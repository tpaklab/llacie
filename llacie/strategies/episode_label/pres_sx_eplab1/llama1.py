from tqdm import tqdm

from ...abstract import AbstractStrategy
from ...feature.presenting_sx.llama1 import PresentingSxFeatureLLama1Strategy
from ....tasks.episode_label import PresentingSymptomsEpisodeLabelV1Task
from ....utils import chunker, echo_info

SECS_IN_24H = 60 * 60 * 24

class PresentingSymptomsEpisodeLabelV1Llama1Strategy(AbstractStrategy):
    """\
    Converts the ..presenting_sx.llama1 note features into episode labels using vocab v1.
    This is the original vocab used for the K08 submission.
    """
    task = PresentingSymptomsEpisodeLabelV1Task
    name = "episode_label.pres_sx_eplab1.llama1"
    version = "0.0.1"
    prereq_tasks = []

    BATCH_SIZE = 1000


    def __init__(self, db, config, **options):
        super().__init__(db, config, **options)
        self.feat_strat = PresentingSxFeatureLLama1Strategy(db, config, **options)


    def run(self, all_episode_ids, batch_size=None):
        fail_count = 0
        vocab = self.task.vocab
        batch_size = batch_size if batch_size is not None else self.BATCH_SIZE

        echo_info(f"In batches of {batch_size}")
        needs_labels_pb = tqdm(chunker(all_episode_ids, batch_size))

        for ep_ids in needs_labels_pb:
            needs_labels_pb.set_description(
                f"Creating labels ({fail_count} failures/{len(all_episode_ids)} total)")
            
            df = self.db.get_earliest_notes_with_feature(ep_ids, self.feat_strat, SECS_IN_24H)
            # Some notes may not have any notes with the required feature
            fail_count += len(ep_ids) - len(df)

            for _, row in df.iterrows():
                labels_dict = vocab.find_terms_in_feature(row.feature_value)

                if len(labels_dict) == 0: 
                    fail_count += 1
                else:
                    self.db.replace_episode_labels(self, row, labels_dict)

        echo_info(f"Finished! ({fail_count} failures/{len(all_episode_ids)} total)")
