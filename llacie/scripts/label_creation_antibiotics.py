import openpyxl
from pathlib import Path
from tqdm import tqdm
import medspacy

SEPARATOR = '######################################'


def insert_into_workbook(all_notes, all_labels, select_sections, output_path: Path):
    wb = openpyxl.Workbook()
    page = wb.active
    page.cell(row=1, column=1, value='ID_Number')
    page.cell(row=1, column=2, value='HPI and Observation sections')
    page.cell(row=1, column=3, value='Labels')
    page.cell(row=1, column=4, value='Entire Note')

    for i, (note, label, section) in enumerate(zip(all_notes, all_labels, select_sections), start=2):
        page.cell(row=i, column=1, value=i)
        page.cell(row=i, column=4, value=note)
        page.cell(row=i, column=3, value=label)
        page.cell(row=i, column=2, value=section)

    wb.save(filename=output_path)


def get_all_notes(pathname: Path):
    notes_list = []
    with open(pathname, 'r') as f:
        current_lines = []
        for line in f:
            if line.strip() == SEPARATOR:
                if current_lines:
                    notes_list.append('\n'.join(current_lines))
                    current_lines = []
            else:
                current_lines.append(line.strip())
        if current_lines:
            notes_list.append('\n'.join(current_lines))
    return notes_list


def get_all_labels(pathname: Path):
    labels = []
    with open(pathname, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                if '. ' in line:
                    _, label = line.split('. ', 1)
                    labels.append(label)
                else:
                    labels.append(line)
    return labels


_nlp = None

def _build_nlp():
    nlp = medspacy.load(enable=["medspacy_pyrush", 'target_matcher'])
    nlp.add_pipe("medspacy_sectionizer")
    return nlp


def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = _build_nlp()
    return _nlp


def extract_medicine_list(note_text):
    categories_list = ['history_of_present_illness', 'observation_and_plan']
    nlp = get_nlp()
    if note_text is None:
        raise ValueError("note_text cannot be None")
    doc = nlp(note_text)
    for sec in doc._.sections:
        if sec.category in categories_list:
            return doc[sec.body_start:sec.body_end].text.strip(":?-_ \xa0\n")
    return None


def run(all_notes):
    fail_count = 0
    results = []
    for note_text in tqdm(all_notes, desc="Extracting medicine sections"):
        extracted = extract_medicine_list(note_text)
        if extracted is not None and len(extracted) > 0:
            results.append(extracted)
        else:
            fail_count += 1
            results.append(None)
    print(f"Finished! ({fail_count} failed/{len(all_notes)} total)")
    return results


def main():
    all_notes = get_all_notes(Path('examples/antibiotics/synthetic_antibiotics_notes.txt'))
    all_labels = get_all_labels(Path('examples/antibiotics/synthetic_antibiotics_labels.txt'))
    select_sections = run(all_notes)
    output_path = Path('examples/antibiotics/synthetic_antibiotics_labels-labels.xlsx')
    insert_into_workbook(all_notes, all_labels,select_sections, output_path)


if __name__ == '__main__':
    main()

