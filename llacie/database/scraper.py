import requests
from bs4 import BeautifulSoup
import pandas as pd
import lxml

def get_page(page_link):
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    }
    request =  requests.get(page_link,headers=headers)
    page_contents = request.text
    return page_contents

def clean_the_document(raw_page):
    soup = BeautifulSoup(raw_page, 'html.parser')
    return soup

def get_the_clinical_data(soup) -> dict | str:
    """ Searches for the wikipedia page"""
    
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
            print('B',b.get_text(strip=True))
            print(row.find('td').get_text())
            if b.get_text(strip=True) == 'Trade names':
                    result['Trade names'] = row.find('td').get_text()
            if b.get_text(strip=True) == 'Other names':
                    result['Other names'] = row.find('td').get_text()
    print(result)
    if result:
        return(result)
    return ("ERROR: DATA NOT FOUND")

def get_and_clinical_data_to_text_file(drug_name):
    print(f'Drug Name: {drug_name}')
    contents = get_page(f"https://en.wikipedia.org/wiki/{drug_name}")
    soup = clean_the_document(contents)
    brand_data= get_the_clinical_data(soup)
    print(f'{drug_name}')
    print(brand_data)

    with open("llacie/database/temp.txt", "a") as f:
        if type(brand_data) !=str:
            f.write(f'{drug_name} | {brand_data["Trade names"]} | {brand_data["Other names"]}')
        else:
            f.write(f'{drug_name} | {brand_data}')
        f.write('\n')


if __name__ == '__main__':
    get_and_clinical_data_to_text_file('Phenoxymethylpenicillin')