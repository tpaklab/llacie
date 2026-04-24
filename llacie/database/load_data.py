import scraper
import pandas 
import random
import time 

def load_data_from_wikipedia(df):
    for item in df['antibioticConceptCode']:
        sleep_time = random.randint(4,6)
        scraper.get_and_clinical_data_to_text_file(item)
        time.sleep(sleep_time)

