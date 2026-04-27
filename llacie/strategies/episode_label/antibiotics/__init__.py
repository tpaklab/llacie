from tqdm import tqdm
import pandas as pd
import json
import re
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
                labels_dict = []
                json_object = json.loads(row['feature_value'])
                for json_parsed in json_object:
                    drug_name = json_parsed['Drug Name'].lower()
                    # Refactor logic to calculate end/start data into new function.
                    diff = get_duration(json_parsed)
                    try:
                        drug_name_label = query(drug_name)[0][0]
                    except Exception as e:
                        drug_name_label = drug_name
                        print(e)
                    labels_dict.append((drug_name_label,diff))
                if len(labels_dict) == 0: 
                    fail_count += 1
                else:
                    self.db.replace_antibiotic_episode_labels(self, row, labels_dict)

        echo_info(f"Finished! ({fail_count} failures/{len(all_episode_ids)} total)")

def get_duration(json_parsed):
    """ Gets the json object and extracts the duration in the las """
    if json_parsed['End date'] is None or json_parsed['Start date'] is None \
         or json_parsed['Start date'] == 'NULL' or json_parsed['End date'] == 'NULL':
        diff = None
    else:
        json_parsed['End date'] = pd.to_datetime(json_parsed['End date'])
        json_parsed['Start date'] = pd.to_datetime(json_parsed['Start date'])
        diff = json_parsed['End date'] - json_parsed['Start date']
        diff = str(diff.days)
        return diff