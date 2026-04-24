import scraper
import pandas  as pd
import random
import time 
from pathlib import Path
def load_data_from_wikipedia(df):
    for item in df['antibioticConceptCode']:
        sleep_time = random.randint(4,6)
        scraper.get_and_clinical_data_to_text_file(item)
        time.sleep(sleep_time)

def load_from_csv_to_xlsx():
    df = pd.read_csv(Path("llacie/database/use_this_data/raw.txt"),sep='|',header=0,)
    df = df.apply(lambda x : x.str.strip() if x.dtype == "object" else x)
    print('=========== Trade names============')
    print(df)
    print('The headers are: ', list(df))
    df.to_excel('llacie/database/use_this_data/cleaned.xlsx',sheet_name='Brand Names',columns=['Drug Name', 'Brand Names'])

if __name__ == '__main__':
    load_from_csv_to_xlsx()()
