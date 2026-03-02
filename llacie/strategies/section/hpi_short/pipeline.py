import re
from tqdm import tqdm
import medspacy
from medspacy.section_detection import Sectionizer, SectionRule

from ...abstract import AbstractStrategy
from ....tasks.section import ShortHPISectionTask
from ....utils import chunker, echo_info

def _build_nlp():
    nlp = medspacy.load(enable=["medspacy_pyrush"])
    nlp.add_pipe("medspacy_sectionizer")
    sectionizer = nlp.get_pipe("medspacy_sectionizer")
    # sectionizer.add([
    #     # HPI start markers
    #     SectionRule("HPI", "hpi_short", pattern=(
    #         r"\b(HPI"
    #         r"|(History|Central[ ]Elements)[ ]of[ ](the[ ])?(Present(ing)?[ ]Illness|Traumatic[ ]Injury)"
    #         r"|Brief[ ]summary)\b"
    #         r"(:|\s+Comments:\s*|\s+HPI\b|\s+History[ ]of[ ](the[ ])?Present[ ]Illness|\s+Chief[ ]Complaint:[^\n]*)?"
    #     )),

    #     # End-of-HPI markers — terminate the HPI body when the next section is detected
    #     SectionRule("Review of Systems",   "ros",         pattern=r"\b(Review[ ]of[ ]Systems|ROS)\b"),
    #     SectionRule("Past Medical History","pmh",         pattern=r"\b(Past[ ](Medical[ ])?(History|Hx)|PMH)\b"),
    #     SectionRule("Medical History",     "pmh",         pattern=r"\bMedical([/\s]+Surgical|[ ]+(and|&)[ ]+Surgical)?[ ](History|Hx)\b"),
    #     SectionRule("ED Course",           "ed_course",   pattern=r"\b(E[DWR]|Emergency[ ](Department|Room))[ ]Course\b"),
    #     SectionRule("Vitals",              "vitals",      pattern=r"\b(ED[ ]Triage[ ])?(Vitals|Vital[ ]Signs)\b"),
    #     SectionRule("Assessment and Plan", "assessment",  pattern=r"\b((Impression|Assessment)[ ]and[ ])?Plan\b|\bA[/&]P\b"),
    #     SectionRule("Physical Exam",       "exam",        pattern=r"\b(Relevant|Pertinent[ ])?(Physical[ ])?Exam\b"),
    #     SectionRule("Medications",         "medications", pattern=r"\b(Relevant|Pertinent[ ])?(Home[ ])?Medications\b"),
    #     SectionRule("Data reviewed",       "data_review", pattern=r"\bData[ ]reviewed\b"),
    #     SectionRule("Current Assessment",  "assessment",  pattern=r"\bCurrent[ ]Assessment\b"),
    #     SectionRule("Historical features", "historical",  pattern=r"\bHistorical[ ]features\b"),
    #     SectionRule("EDD",                 "edd",         pattern=r"\b(Estimated[ ]Date[ ]of[ ]Delivery|EDD)\b"),
    #     SectionRule("History provided by", "history_src", pattern=r"\bHistory[ ]provided[ ]by\b"),
    #     SectionRule("EMR Reviewed",        "emr_review",  pattern=r"\bElectronic[ ]Medical[ ]Records[ ]Reviewed\b"),
    #     SectionRule("In the ED",           "ed_arrival",  pattern=r"\b(In|On[ ]arrival[ ](to|at))[ ](the[ ])?(\w+[ ])?(E[DWR]|Emergency[ ](Room|Department))\b"),
    #     SectionRule("Patient Active Problem List", "problem_list",  pattern=r"\bPatient[ ]Active[ ]Problem[ ]List\b"),
    #     SectionRule("Focused COVID History",      "covid_history", pattern=r"\bFocused[ ]COVID[ ]History\b"),
    #     SectionRule("Quality Bundle",             "quality_bundle",pattern=r"\bQuality[ ]Bundle\b"),
    # ])

    return nlp


class ShortHPISectionSpacyStrategy(AbstractStrategy):
    """\
    Attempts to extract the `hpi_short` section using medspaCy's sectionizer
    to identify the HPI header and return everything up to the next section.
    """
    task = ShortHPISectionTask
    name = "section.hpi_short.spacy"
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
    def extract_short_hpi(cls, note_text):
        nlp = cls.get_nlp()
        if note_text is not None:
            print('Passes note-text')
            doc = nlp(note_text)
        else:
            raise FileNotFoundError
        print(doc)
        for sec in doc._.sections:
            print(sec)
            if sec.category in ("hpi_short", "history_of_present_illness"):
                print('Inside HPI category')
                hpi_text = doc[sec.body_start:sec.body_end].text
                print(f"HPI section [{sec.body_start}:{sec.body_end}]:  {hpi_text[:80]!r}")
                print('Extracted',hpi_text.strip(":?-_ \xa0\n"))
                return hpi_text.strip(":?-_ \xa0\n")
                

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
                print('HPI',row['hpi_short'])
                if row['hpi_short'] is not None and len(row['hpi_short']) > 0:
                    print('Insert')
                    self.db.insert_note_section(row['id'], self, row['hpi_short'])
                else:
                    fail_count += 1


        echo_info(f"Finished! ({fail_count} failed/{len(all_note_ids)} total)")
