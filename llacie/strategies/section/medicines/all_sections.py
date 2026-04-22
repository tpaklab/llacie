# All the medspacy categories for reference
# {'problem_list', 'other', 'neurological', 'history_of_present_illness', 'allergies', 
# 'chief_complaint', 'siganture', 'medications', 'diagnoses', 'social_history', 'labs_and_studies', 'patient_education',
#  'observation_and_plan', 'patient_instructions', 'imaging', 'hospital_course',
# 'reason_for_examination', 'signature', 'addendum', 'comments', 'family_history', 'allergy',
#  'past_medical_history', 'physical_exam'}
import re
from tqdm import tqdm
import medspacy
from medspacy.section_detection import Sectionizer, SectionRule
from medspacy.target_matcher import TargetMatcher, TargetRule
from ...abstract import AbstractStrategy
from ....tasks.section import AntibioticsTask
from ....utils import chunker, echo_info



class AntibioticsSpacyStrategy(AbstractStrategy):
    """\
    Attempts to extract the all relevant antibiotic sections using medspaCy's sectionizer
    """
    task = AntibioticsTask
    name = "section.antibiotics.all_sections"
    version = "0.0.1"
    BATCH_SIZE = 1000

    prereq_tasks = []

    _nlp = None



    @staticmethod
    def clean_note_text(text):
        text = re.sub("  ", "\n", text)
        text = re.sub("\n[ ?]+", "\n", text)
        text = re.sub("\n\n+", "\n\n", text)
        return text.strip()

    @classmethod
    def extract_medicine_list(cls, note_text):
        return note_text
    
    def run(self, all_note_ids,max_note_ids=None, batch_size=None,dry_run=None):
        fail_count = 0
        batch_size = batch_size if batch_size is not None else self.BATCH_SIZE

        echo_info(f"In batches of {batch_size}")
        needs_hpi_pb = tqdm(chunker(all_note_ids, batch_size))

        with self.db.foreign_keys_temporarily_dropped('note_sections'):
            for note_ids in needs_hpi_pb:
                needs_hpi_pb.set_description(
                    f"Extracting HPIs ({fail_count} failed/{len(all_note_ids)} total)")
                df = self.db.get_full_text_notes(note_ids)
                df['note_text'] = [self.clean_note_text(text) for text in df['note_text']]
                df['medications']= [self.extract_medicine_list(text) for text in df['note_text']]

            for _, row in df.iterrows():
                if row['medications'] is not None and len(row['medications']) > 0:
                    self.db.insert_note_section(row['id'], self, row['medications'])
                else:
                    fail_count += 1


        echo_info(f"Finished! ({fail_count} failed/{len(all_note_ids)} total)")

