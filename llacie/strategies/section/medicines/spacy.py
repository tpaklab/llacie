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
from ....tasks.section import ShortHPISectionTask
from ....utils import chunker, echo_info

def _build_nlp():
    nlp = medspacy.load(enable=["medspacy_pyrush",'target_matcher'])
    nlp.add_pipe("medspacy_sectionizer")
    sectionizer = nlp.get_pipe("medspacy_sectionizer")

    antibiotic_rules = [
        TargetRule(
        literal="anti-biotics",
        category="MEDICATION",
        pattern=[{"TEXT": {"REGEX": "(?i).*mycin$|.*cillin$|.*floxacin$"}}])

    ]
    
    sectionizer.add(antibiotic_rules)

    return nlp


class ShortHPISectionSpacyStrategy(AbstractStrategy):
    """\
    Attempts to extract the `hpi_short` section using medspaCy's sectionizer
    to identify the HPI header and return everything up to the next section.
    """
    task = ShortHPISectionTask
    name = "section.antibiotics.spacy"
    version = "0.0.1"
    BATCH_SIZE = 1000

    prereq_tasks = []

    _nlp = None

    @classmethod
    def get_nlp(cls):
        if cls._nlp is None:
            cls._nlp = _build_nlp()
        return cls._nlp

    @staticmethod
    def clean_note_text(text):
        text = re.sub("  ", "\n", text)
        text = re.sub("\n[ ?]+", "\n", text)
        text = re.sub("\n\n+", "\n\n", text)
        return text.strip()

    @classmethod
    def extract_medicine_list(cls, note_text):
        categories_list = ['history_of_present_illness',]
        nlp = cls.get_nlp()
        if note_text is not None:
            print('Passes note-text')
            doc = nlp(note_text)
        else:
            raise FileNotFoundError
        print(doc)
        for sec in doc._.sections:
            print(sec)
            if sec.category in categories_list:
                hpi_text = doc[sec.body_start:sec.body_end].text
                print(f"HPI section [{sec.body_start}:{sec.body_end}]:  {hpi_text[:80]!r}")
                return hpi_text.strip(":?-_ \xa0\n")
                e
        return None

    def run(self, all_note_ids, batch_size=None):
        fail_count = 0
        batch_size = batch_size if batch_size is not None else self.BATCH_SIZE

        echo_info(f"In batches of {batch_size}")
        needs_hpi_pb = tqdm(chunker(all_note_ids, batch_size))

        with self.db.foreign_keys_temporarily_dropped('note_sections'):
            num = 0
            for note_ids in needs_hpi_pb:
                needs_hpi_pb.set_description(
                    f"Extracting HPIs ({fail_count} failed/{len(all_note_ids)} total)")
                df = self.db.get_full_text_notes(note_ids)
                df['note_text'] = [self.clean_note_text(text) for text in df['note_text']]
                df['hpi_short']= [self.extract_short_hpi(text) for text in df['note_text']]

            for _, row in df.iterrows():
                if row['hpi_short'] is not None and len(row['hpi_short']) > 0:
                    self.db.insert_note_section(row['id'], self, row['hpi_short'])
                else:
                    fail_count += 1


        echo_info(f"Finished! ({fail_count} failed/{len(all_note_ids)} total)")

