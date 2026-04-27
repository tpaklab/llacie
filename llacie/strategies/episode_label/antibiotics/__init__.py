from tqdm import tqdm
import pandas as pd
import json
from ...abstract import AbstractStrategy
from ....tasks.episode_label import AntibioticsEpisodeLabelTask
from ....utils import chunker, echo_info
from ....database.query_db import query
SECS_IN_24H = 60 * 60 * 24

class AbstractAntibioticsStrategy(AbstractStrategy):
    """\
    An abstract strategy for converting presenting_sx note features into episode labels using 
    vocab v2. This vocab is UPDATED since the K08 submission after manual examination of the
    misclassifications.
    """
    task = AntibioticsEpisodeLabelTask
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
            print('Antibiotic df')
            df = self.db.get_earliest_notes_with_feature(ep_ids, self.feat_strat, SECS_IN_24H)
            #print(df)

            # Some episodes may not have any notes with the required feature
            fail_count += len(ep_ids) - len(df)
        
            for _, row in df.iterrows():
                json_parsed = json.loads(row['feature_value'])[0]
                print(json_parsed)
                drug_name = json_parsed['Drug Name'].lower()
                # FIXME: Change the logic when creating it to N/A instead of NULL
                if json_parsed['End date'] is None or json_parsed['Start date'] is None or json_parsed['Start date'] == 'NULL' or json_parsed['End date'] == 'NULL':
                    diff = None
                else:
                    json_parsed['End date'] = pd.to_datetime(json_parsed['End date'])
                    json_parsed['Start date'] = pd.to_datetime(json_parsed['Start date'])
                    diff = json_parsed['End date'] - json_parsed['Start date']
                    diff = str(diff.days)
                labels_dict = (query(drug_name)[0][0],diff)
                if len(labels_dict) == 0: 
                    fail_count += 1
                else:
                    self.db.replace_antibiotic_episode_labels(self, row, labels_dict)

        echo_info(f"Finished! ({fail_count} failures/{len(all_episode_ids)} total)")

