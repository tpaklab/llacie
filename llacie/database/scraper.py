import requests
from bs4 import BeautifulSoup
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

def get_the_clinical_data(soup):
    
    found_clinical = False
    raw_clinical_data = []
    table = soup.find("table",class_ ="infobox")
    if table is None:
        return ("ERROR: DATA NOT FOUND", "ERROR")

    for x in table.find_all("tr"):
        if 'Clinical data' in x.get_text(strip = True):
            found_clinical = True
            continue
        if found_clinical:
            th = x.find('th')
            if th and th.get('colspan'):
                break
            raw_clinical_data.append(x)
    for row in raw_clinical_data:
        a=row.find('a')
        if a:
            if a.get_text(strip=True) == 'Trade names':
                return (a.get_text(),row.find('td').get_text())
    return ("ERROR: DATA NOT FOUND", "ERROR")

def get_and_clinical_data_to_text_file(drug_name):
    print(f'Drug Name: {drug_name}')
    contents = get_page(f"https://en.wikipedia.org/wiki/{drug_name}")
    soup = clean_the_document(contents)
    brand_data= get_the_clinical_data(soup)

    with open("llacie/database/temp.txt", "a") as f:
        f.write(f'{drug_name} | {brand_data[0]} | {brand_data[1]}')
        f.write('\n')


if __name__ == '__main__':
    NotImplementedError
