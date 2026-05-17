## This file is called from main.py
## It saves every scraped page into raw_wiki.txt
## For any small changes like drug addition, just add it manually to raw_wiki.txt
## If you need to scrape all wikipedia pages again, then uncomment the line in main.py's main()

import requests
from bs4 import BeautifulSoup
import pandas as pd
import random
import time
from pathlib import Path

def read_vocab(file_path:Path="llacie/database/files/micro_antibiotics_TRP.xlsx")-> pd.DataFrame:
    """ Reads `micro_antibiotics_TRP.xlsx` file and loads into pandas."""
    df = pd.read_excel(Path(file_path),sheet_name="antibiotic_map")
    df = df.drop_duplicates(subset=['antibioticConceptCode'])
    df = df.dropna()
    return df


def get_page(page_link):
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    }
    request =  requests.get(page_link,headers=headers)
    page_contents = request.text
    return page_contents

def parse_page(soup) -> dict | str:
    """ Parses a Wikipedia beautifulsoup object to get a drug's trade and other names. 
        Returns a dict, returns str when error in parsing. """
    
    found_clinical = False
    raw_clinical_data = []
    table = soup.find("table",class_ ="infobox")
    if table is None:
        print("The webpage was not found")
        return ("ERROR: DATA NOT FOUND")

    for x in table.find_all("tr"):
        if 'Clinical data' in x.get_text(strip = True):
            found_clinical = True
            continue
        if found_clinical:
            th = x.find('th')
            if th and th.get('colspan'):
                break
            raw_clinical_data.append(x)
    result = {}
    result['Trade names'] = None
    result['Other names'] = None
    for row in raw_clinical_data:
        b = row.find('th')
        if b:
            if b.get_text(strip=True) == 'Trade names':
                    result['Trade names'] = row.find('td').get_text()
            if b.get_text(strip=True) == 'Other names':
                    result['Other names'] = row.find('td').get_text()
    if result:
        return(result)
    return ("ERROR: DATA NOT FOUND")

def get_data_and_save(drug_name):
    """ Given a `drug name`, gets drug data and saves it to `raw_wiki.txt` """
    print(f'Drug Name: {drug_name}')
    contents = get_page(f"https://en.wikipedia.org/wiki/{drug_name}")
    if not contents:
        print(f'{drug_name} has no wiki page')
    soup = BeautifulSoup(contents, 'html.parser')
    brand_data= parse_page(soup)
    with open("llacie/database/files/raw_wiki.txt", "a") as f:
        if type(brand_data) !=str:
            f.write(f'{drug_name} | {brand_data["Trade names"]} | {brand_data["Other names"]}')
        else:
            f.write(f'{drug_name} | {brand_data}')
        f.write('\n')

def load_data_from_wikipedia(df):
    for item in df['antibioticConceptCode']:
        sleep_time = random.randint(4,6)
        get_data_and_save(item)
        time.sleep(sleep_time)


if __name__ == '__main__':

