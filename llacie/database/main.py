import pandas as pd
import numpy as np
from pathlib import Path
import load_data as ld

def load_data_into_pandas()-> pd.DataFrame:
    df = pd.read_excel(Path("llacie/database/micro_antibiotics_TRP.xlsx"),sheet_name="antibiotic_map")
    df = df.drop_duplicates(subset=['antibioticConceptCode'])
    df = df.dropna()
    print(df)
    return df


def main():
    df  = load_data_into_pandas()
    ld.load_data_from_wikipedia(df)

    
if __name__ == '__main__':
    main()
