import numpy as np
import pandas as pd
import confidenceinterval as ci

import click 

from math import log10
from scipy import stats
from sklearn import metrics
from textwrap import dedent
from collections import OrderedDict

pd.set_option('future.no_silent_downcasting', True)

def filter_to_one_humans_rows_per_episode(df):
    """For each set of rows in a table, find the first human annotator per FK_episode_id,
    and only return their responses.
    
    Intended for use with the episode_labels table."""
    first_humans = df.groupby('FK_episode_id')['FK_human_annotator'].first().reset_index()

    return pd.merge(df, first_humans, on=['FK_episode_id', 'FK_human_annotator'], how='inner')


def create_vocab_matrix(df, vocab):
    """Creates a dataframe with a column for every term in `vocab`, and one row for each
    row in `df`. Any existing columns are preserved; any new columns are filled with False.
    The columns are sorted in alphabetical order."""
    cols_needed = [term for term in vocab.terms if term not in df]
    data = np.full((len(df), len(cols_needed)), False)
    df_empty = pd.DataFrame(data, columns=cols_needed, index=df.index)
    return pd.concat([df, df_empty], axis=1)[vocab.terms]


def f1_score_binary(y_true, y_pred):
    return ci.f1_score(y_true, y_pred, average='binary')


class ConfusionMatrix:
    # If these are a function, they should accept `y_true` and `y_pred` as arguments, similar to
    # the metrics in scikit-learn or the score functions in the `confidenceinterval` package
    # If these are a string, the corresponding method of the instance of this class is called
    #   (this allows instances to have attributes that affect the behavior of the metric)
    METRIC_METHODS = {
        "Accuracy": ci.accuracy_score,
        "Balanced accuracy": "_balanced_accuracy_score",
        "F1 score": f1_score_binary,
        "Precision aka PPV": ci.ppv_score,
        "Recall aka sensitivity": ci.tpr_score,
        "Specificity": ci.tnr_score,
        "NPV": ci.npv_score,
        "Cohen's kappa": "_cohen_kappa_score",
        "Spearman-R":'_spearman_r', 
        "Wilcox-Assigned-Rank":"_wilcox_assigned_rank"
    }
    DEFAULT_N_RESAMPLES = 1000
    METRIC_PRECISION_SCALING = 2.4
    METRIC_MIN_DIGITS = 3

    def __init__(self, vec_truth, vec_pred, value_true, value_pred,other_human=None, n_resamples=None):
        self._vec_truth = vec_truth
        self._vec_pred = vec_pred
        self._value_true = value_true
        self._value_pred = value_pred
        self._matrix = metrics.confusion_matrix(vec_truth, vec_pred)
        self._tn, self._fp, self._fn, self._tp = self._matrix.ravel()
        self._other_human = other_human

        # For metrics with bootstrapped confidence intervals
        self._n_resamples = self.DEFAULT_N_RESAMPLES if n_resamples is None else n_resamples
        self._random_generator = np.random.default_rng()
    
    @property
    def matrix(self): return self._matrix

    @property
    def tn(self): return self._tn

    @property
    def tp(self): return self._tp

    @property
    def fn(self): return self._fn

    @property
    def fp(self): return self._fp

    @classmethod
    def from_episode_labels(cls, df_truth, df_pred, vocab, max_line_num=9, **kwargs):
        df_truth = filter_to_one_humans_rows_per_episode(df_truth)

        merged_values = pd.merge(
            df_truth[['FK_episode_id', 'label_name', 'label_value', 'line_number']],
            df_pred[['FK_episode_id', 'label_name', 'label_value','line_number']],
            on=['FK_episode_id', 'label_name', 'line_number'],
            suffixes=('_true', '_pred')
        )
        value_true = merged_values['label_value_true']
        value_pred = merged_values['label_value_pred']

        df_truth = df_truth[["FK_episode_id", "label_name", "line_number"]].copy()
        df_truth["labeled"] = True
        df_truth_pivot = df_truth.pivot(index="FK_episode_id", columns="label_name", 
                values="labeled")
        df_truth_pivot.fillna(False, inplace=True)
        df_truth_mat = create_vocab_matrix(df_truth_pivot, vocab)

        df_pred = df_pred[["FK_episode_id", "label_name", "line_number"]].copy()
        df_pred["labeled"] = df_pred["line_number"] <= max_line_num
        # print(df_pred.to_string())


        try:
            df_pred_pivot = df_pred.pivot(index="FK_episode_id", columns="label_name", 
                values="labeled")
        except ValueError as e:
            print(e)

        df_truth_ep_ids = df_truth_pivot.reset_index()[["FK_episode_id"]]
        df_pred_pivot = pd.merge(df_truth_ep_ids, df_pred_pivot, 
            on="FK_episode_id", how="left", suffixes=('_X', ''))
        df_pred_pivot.fillna(False, inplace=True)
        df_pred_pivot.index = df_truth_ep_ids["FK_episode_id"]
        df_pred_mat = create_vocab_matrix(df_pred_pivot, vocab)

        vec_truth = df_truth_mat.values.flatten().astype(bool)
        vec_pred = df_pred_mat.values.flatten().astype(bool)


        return cls(vec_truth, vec_pred, value_true, value_pred ,**kwargs)


    def _bootstrapped_metric(self, y_true, y_pred, metric):
        return ci.bootstrap.bootstrap_ci(
            y_true=y_true,
            y_pred=y_pred,
            metric=metric,
            confidence_level=0.95,
            n_resamples=self._n_resamples,
            method='bootstrap_percentile',
            random_state=self._random_generator
        )

    def _balanced_accuracy_score(self, y_true, y_pred):
        return self._bootstrapped_metric(y_true, y_pred, metrics.balanced_accuracy_score)


    def _cohen_kappa_score(self, y_true, y_pred):
        return self._bootstrapped_metric(y_true, y_pred, metrics.cohen_kappa_score)

    def _spearman_r(self, y_true, y_pred):
        y_true.fillna(0, inplace=True)
        y_pred.fillna(0, inplace=True)
        return stats.spearmanr(y_true, y_pred)

    def _count_expected_Nones(self, y_true, y_pred):
        counts=0
        n = 0
        for i,val in enumerate(y_true):
            if val == None:
                if y_pred[i] == None:
                    counts+=1
                n+=1
        return counts // n

    def _wilcox_assigned_rank(self,y_true, y_pred):
        y_true.fillna(0, inplace=True)
        y_pred.fillna(0, inplace=True)
        return stats.wilcoxon(y_true, y_pred)

    def metrics(self):
        time_methods = set(['Spearman-R','Wilcox-Assigned-Rank'])
        ret = OrderedDict()
        for metric, method in self.METRIC_METHODS.items():
            # Most metric methods are stateless functions, but some are dynamically obtained from
            #   this instance in order to capture external parameters (e.g., n_resamples)
            if not callable(method): method = getattr(self, method)
            # Metric methods expect two vectors, one for ground-truth and one for predictions
            # We now use methods from https://github.com/jacobgil/confidenceinterval
            # These are similar to sklearn.metrics but also provide a confidence interval 
            #   along with the point estimate
            if metric in time_methods:
                ret[metric] = method(self._value_true, self._value_pred)
            else:
                ret[metric] = method(self._vec_truth, self._vec_pred)
        return ret
    

    def _format_metric(self, metric_name, metric_value):
        if hasattr(metric_value, 'statistic'):  # SpearmanrResult
            return f"{metric_name:23.23} {metric_value.statistic:10.4f} (p={metric_value.pvalue:.4f})"
        if isinstance(metric_value, tuple):
            prec = int(-log10(1 - metric_value[0]) + self.METRIC_PRECISION_SCALING)
            prec = max(prec, self.METRIC_MIN_DIGITS)
            # Metric includes a confidence interval
            metric_and_ci = (f"{metric_value[0]:10.{prec}f} ({metric_value[1][0]:9.{prec}f}, "
                f"{metric_value[1][1]:9.{prec}f})")
            return f"{metric_name:23.23} {metric_and_ci}"
        else:
            return f"{metric_name:23.23} {metric_value:10.2f}"


    def __str__(self):
        metrics = ("--------------------------- Metric ( 95% CI ) -------------\n" +
            "\n".join([self._format_metric(key, val) for key, val in self.metrics().items()])) 
        # + 
        #     "\n".join('Mean absolute Error:')

        header_1 = "                           Ground truth labels"
        header_2 = "                           Present     Absent"
        whose = "Predicted" if self._other_human is None else f"{self._other_human}'s"
        conf_mat = dedent(f"""\
            {header_1}
            {header_2}
            {whose+" label present":23} {str(self.tp):>10} {str(self.fp):>10}
            {whose+" label absent":23} {str(self.fn):>10} {str(self.tn):>10}""") 
        
        return conf_mat + "\n" + metrics