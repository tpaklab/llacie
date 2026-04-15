import xml.etree.ElementTree as ET
import query_drugbank as query

NS = "http://www.drugbank.ca"


def push_into_db(conn):
    context = ET.iterparse('llacie/database/full database.xml', events=("end",))

    for _, element in context:
        if element.tag == f'{{{NS}}}drug':
            process_element(element, conn)
            element.clear()

    conn.commit()


def process_element(element, conn):
    drug_id = None
    all_ids = element.findall(f"{{{NS}}}drugbank-id")
    for did in all_ids:
        if did.get("primary") in ("true", "1"):
            drug_id = did.text
            break
    if drug_id is None and all_ids:
        drug_id = all_ids[0].text
    name = element.findtext(f"{{{NS}}}name")
    if drug_id and name:
        query.insert_drug(conn, drug_id, name)


if __name__ == "__main__":
    conn = query.intialize_drugbank()
    push_into_db(conn)







