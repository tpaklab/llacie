import re
import pickle
import platformdirs
import pandas as pd

from os import path, makedirs
from collections import defaultdict
from copy import deepcopy

from .utils import echo_info, echo_warn, PACKAGE_DIR


class Vocab:
    """Implements a vocabulary containing terms, used for labeling tasks. The tabular
    form of each vocabulary is stored in XLSX files under vocabs/. See those files for
    examples of how to format them.
    
    This class uses a pickle file (cache) to avoid reparsing the XLSX whenever possible."""

    def __init__(self, from_file, sheet_name=0, **options):
        self._terms = defaultdict(set)
        self._ngram_dicts = []

        self.from_file = path.join(PACKAGE_DIR, "vocabs", from_file)
        cache_dir = platformdirs.user_cache_dir("llacie", "tpaklab")
        self.cache_file = path.join(cache_dir, "vocabs", f"{from_file}.pkl")

        if self._load_from_cache():
            return
        elif self.from_file.endswith(".xlsx") or self.from_file.endswith(".xls"):
            echo_info(f"Parsing and caching vocabulary in vocabs/{from_file}")
            self._parse_df(pd.read_excel(self.from_file, sheet_name=sheet_name))
        else:
            raise NotImplementedError

    def __contains__(self, term):
        return term in self._terms

    def __len__(self):
        return len(self._terms)
    

    @property
    def terms(self):
        """Returns a list of all (canonical) terms in this vocab, omitting synonyms."""
        return sorted(self._terms.keys())
    
    @property
    def terms_and_synonyms(self):
        """Returns a dict of terms mapped to corresponding lists of synonyms. This is a copy,
        so modifying it has no effect on this Vocab instance."""
        return deepcopy(self._terms)


    def _load_from_cache(self):
        if not path.isfile(self.cache_file): return False
        if path.getsize(self.cache_file) == 0: return False
        if path.getmtime(self.cache_file) < path.getmtime(self.from_file): return False
        try:
            with open(self.cache_file, 'rb') as f:
                loaded_obj = pickle.load(f)
                self._terms = loaded_obj["_terms"]
                self._ngram_dicts = loaded_obj["_ngram_dicts"]
        except (pickle.UnpicklingError, PermissionError, TypeError, KeyError) as e:
            echo_warn(f"Error loading cached vocab: {e}")
            return False
        return True


    def _save_to_cache(self):
        makedirs(path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'wb') as f:
            pickled = {"_terms": self._terms, "_ngram_dicts": self._ngram_dicts}
            pickle.dump(pickled, f)


    def _add_terms(self, terms, synonyms=None):
        if synonyms is not None:
            if not isinstance(synonyms, list):
                synonyms = [synonyms]
        for term in terms:
            self._terms[term].add(term)
            self._terms[term].update(synonyms)


    def _parse_df(self, vocab_df):
        max_n = max(vocab_df['n'])
        for n in range(max_n, 0, -1):
            ngram_dict = {}
            for _, row in vocab_df[vocab_df.n == n].iterrows():
                terms = [row['canonical_name']]
                if pd.notna(row['combo_symptom']):
                    terms.append(row['combo_symptom'])
                self._add_terms(terms, row['ngram'])
                ngram_dict[tuple(row['ngram'].split(' '))] = terms
            self._ngram_dicts.append(ngram_dict)
        self._save_to_cache()


    def find_terms_in_feature(self, feature_value):
        """Given a textual feature, searches each line of the feature for terms from this
        vocabulary, preferring longer n-gram matches over shorter ones. This is akin to the
        backoff techniques used in NLP: https://www.scaler.com/topics/nlp/backoff-in-nlp/
        
        Returns a dict() with matched terms as keys and earliest line # matched as values."""

        lines = feature_value.split("\n")
        found_terms = {}
        for line_no in range(len(lines) - 1, -1, -1):
            tokens = re.split(r'\s+', re.sub(r'[^a-z0-9]+', ' ', lines[line_no].lower()).strip())
            while len(tokens) > 0:
                for ngram_i, ngram_dict in enumerate(self._ngram_dicts):
                    n = len(self._ngram_dicts) - ngram_i
                    vocab_matches = ngram_dict.get(tuple(tokens[0:n]), None)
                    if vocab_matches is None: continue
                    for match in vocab_matches:
                        found_terms[match] = line_no + 1
                    tokens = tokens[n:]
                    break
                if vocab_matches is None: tokens.pop(0)
        return found_terms