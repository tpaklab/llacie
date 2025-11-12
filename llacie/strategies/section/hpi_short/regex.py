import re
from tqdm import tqdm

from ...abstract import AbstractStrategy
from ....tasks.section import ShortHPISectionTask
from ....utils import chunker, echo_info

class ShortHPISectionRegexStrategy(AbstractStrategy):
    """\
    Attempts to extract the `hpi_short` section using regular expressions
    to find most likely beginning and ending delimiters within each note.
    """
    task = ShortHPISectionTask
    name = "section.hpi_short.regex"
    version = "0.0.1"
    BATCH_SIZE = 1000

    prereq_tasks = []

    # Markers for the start of the HPI
    START_REGEX = re.compile(r"""
        \b(
            HPI
            | (History|Central[ ]Elements)[ ]of[ ](the[ ])?(Present(ing)?[ ]Illness|Traumatic[ ]Injury)
            | Brief[ ]summary
        )\b( # Optional prefixes within the HPI text that should be excised
            :
            | \s+Comments:\s*
            | \s+HPI\b
            | \s+History[ ]of[ ](the[ ])?Present[ ]Illness
            | \s+Chief[ ]Complaint:[^\n]*
        )*
        """, re.VERBOSE | re.IGNORECASE)

    # Note that for short HPIs, we will be aggressive about trimming into ED and hospital courses
    # at the end of longer HPIs. Short HPI aims to capture the "meat", often just the first paragraph.
    END_REGEX = re.compile(r"""
        (
            \n ( 
                # Markers for end of HPI when it *begins* a line, w/ exact terminal punctuation
                History[ ]reviewed.[ ]+No[ ]pertinent[ ]past[ ]medical[ ]history.
            )
            | \n (
                # Same, but these terms end with a colon or newline instead of specific punctuation
                # This implies that the entire phrase was "formatted" as a heading or subheading
                Electronic[ ]Medical[ ]Records[ ]Reviewed
                | History[ ]provided[ ]by
                | (E[DWR]|Emergency[ ](Department|Room))[ ]Course
                | Current[ ]Assessment
                | Historical[ ]features
                | Focused[ ]COVID[ ]History
                | (ED[ ]Triage[ ])? (Vitals|Vital[ ]Signs)
                | (Plan|A[/&]P)
                | (Estimated[ ]Date[ ]of[ ]Delivery|EDD)
                | (Relevant|Pertinent[ ])? (Home[ ])? Medications
                | (Relevant|Pertinent[ ])? (Physical[ ])? Exam
                | Quality[ ]Bundle
            ) [:\n]
            | \n (
                # Finally these terms can end on *any word boundary*
                # These words imply high confidence for end of HPI, even when beginning a sentence
                Review[ ]of[ ]Systems
                | ROS
                | Past[ ](Medical[ ])?(History|Hx)
                | Patient[ ]Active[ ]Problem[ ]List
                | Medical (([/\s]+|[ ]+(\band\b|&)[ ]+)Surgical)? [ ](History|Hx)
                | PMH
                | (In[ ]|On[ ]arrival[ ](to|at)[ ]) (the[ ])? (\w+[ ])? 
                    (E[DWR]|Emergency[ ](Room|Department))
                | Data reviewed
                | ((Impression|Assessment)[ ]and[ ]) Plan
            ) \b
        )
        """, re.VERBOSE | re.IGNORECASE)


    @staticmethod
    def clean_note_text(text):
        text = re.sub("  ", "\n", text)
        text = re.sub("\n[ ?]+", "\n", text)
        text = re.sub("\n\n+", "\n\n", text)
        return text.strip()
    

    @classmethod
    def extract_short_hpi(cls, note_text, debug=False):
        start_token = re.search(cls.START_REGEX, note_text)
        if start_token is None: return None
        start_token_end = start_token.end(0)

        end_token = re.search(cls.END_REGEX, note_text[start_token_end:None])
        end_token_start = end_token.start(0) + start_token_end if end_token is not None else None
        if debug: print(start_token, end_token)

        hpi_text = note_text[start_token_end:end_token_start]
        # \xa0 = non-breaking space
        return hpi_text.strip(":?-_ \xa0\n")  


    def run(self, all_note_ids, batch_size=None):
        fail_count = 0
        batch_size = batch_size if batch_size is not None else self.BATCH_SIZE

        echo_info(f"In batches of {batch_size}")
        needs_hpi_pb = tqdm(chunker(all_note_ids, batch_size))

        # Temporarily drop foreign keys during this operation.
        # This speeds up inserts by 10x or more; we don't expect perfect ACID during every task
        with self.db.foreign_keys_temporarily_dropped('note_sections'):
            for note_ids in needs_hpi_pb:
                needs_hpi_pb.set_description(
                    f"Extracting HPIs ({fail_count} failed/{len(all_note_ids)} total)")
                
                df = self.db.get_full_text_notes(note_ids)
                df['note_text'] = [self.clean_note_text(text) for text in df['note_text']]
                df['hpi_short'] = [self.extract_short_hpi(text) for text in df['note_text']]

                for _, row in df.iterrows():
                    if row['hpi_short'] is not None and len(row['hpi_short']) > 0: 
                        self.db.insert_note_section(row['id'], self, row['hpi_short'])
                    else: 
                        fail_count += 1
        
        echo_info(f"Finished! ({fail_count} failed/{len(all_note_ids)} total)")
