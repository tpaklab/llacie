import pandas as pd
from pathlib import Path


def extract_rows_and_columns(input_file:Path) -> pd.DataFrame:
    rows = []
    with open(Path(input_file), 'r') as file:
        for line in file:
            if line.startswith('note_'):
                rows.append([x.strip() for x in line.split('|') ])

    df = pd.DataFrame(rows[1:],columns=rows[0])
    return df


def export_dataframe_to_excel(dataframe:pd.DataFrame,excel_path:str)-> bool:
    """ Exports a dataframe to Excel"""
    dataframe.replace("NULL", pd.NA, inplace=True)
    dataframe.to_excel(excel_path)
    return True


def main():
    df = extract_rows_and_columns(Path('examples/antibiotics/synthetic_antibiotics_table_realistic.txt'))
    export_dataframe_to_excel(df,Path('examples/antibiotics/synthetic_antibiotics_table_realistic.xlsx'))

if __name__ == '__main__':
    main()




 