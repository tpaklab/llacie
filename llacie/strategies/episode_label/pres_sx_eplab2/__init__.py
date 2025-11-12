from tqdm import tqdm

from ...abstract import AbstractStrategy
from ....tasks.episode_label import PresentingSymptomsEpisodeLabelV2Task
from ....utils import chunker, echo_info

SECS_IN_24H = 60 * 60 * 24

class AbstractPresentingSymptomsEpisodeLabelV2Strategy(AbstractStrategy):
    """\
    An abstract strategy for converting presenting_sx note features into episode labels using 
    vocab v2. This vocab is UPDATED since the K08 submission after manual examination of the
    misclassifications.
    """
    task = PresentingSymptomsEpisodeLabelV2Task
    prereq_tasks = []

    BATCH_SIZE = 1000


    def run(self, all_episode_ids, batch_size = None):
        fail_count = 0
        vocab = self.task.vocab
        batch_size = batch_size if batch_size is not None else self.BATCH_SIZE

        echo_info(f"In batches of {batch_size}")
        needs_labels_pb = tqdm(chunker(all_episode_ids, batch_size))

        for ep_ids in needs_labels_pb:
            needs_labels_pb.set_description(
                f"Creating labels ({fail_count} failures/{len(all_episode_ids)} total)")
            
            df = self.db.get_earliest_notes_with_feature(ep_ids, self.feat_strat, SECS_IN_24H)
            # Some episodes may not have any notes with the required feature
            fail_count += len(ep_ids) - len(df)

            for _, row in df.iterrows():
                labels_dict = vocab.find_terms_in_feature(row.feature_value)

                if len(labels_dict) == 0: 
                    fail_count += 1
                else:
                    self.db.replace_episode_labels(self, row, labels_dict)

        echo_info(f"Finished! ({fail_count} failures/{len(all_episode_ids)} total)")
