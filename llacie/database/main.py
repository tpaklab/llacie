import re
import pandas as pd
import sqlite3 as sql3
import scraper as sc


def create_db()->sql3.Connection:
    create_drugs_table = """\
    CREATE TABLE IF NOT EXISTS drugs(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cannonical_name TEXT UNIQUE
    );
    """
    create_synonyms_table = """\
    CREATE TABLE IF NOT EXISTS synonyms(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    drug_id INTEGER REFERENCES drugs(id),
    other_names TEXT UNIQUE
    );
    """ 
    conn = sql3.connect("llacie/database/drugs.db")
    # conn.set_trace_callback(print)
    c = conn.cursor()
    c.execute(create_drugs_table)
    c.execute(create_synonyms_table)
    return conn


def _clean_name_list(val):
    if pd.isna(val) or str(val).strip().lower() in ('none', 'error: data not found'):
        return []
    val = re.sub(r'\[\d+\](?::\s*\d+)?', '', val)       # [1], [2]: 185
    val = re.sub(r'\([^)]*\)', '', val)                   # (BAN UK), (USAN US)
    val = re.sub(r',?\s*others?\b', '', val, flags=re.IGNORECASE)
    parts = re.split(r'[,;]', val)
    parts = [p.strip().lower() for p in parts if p.strip()]
    # drop long IUPAC names: stereochemistry notation + length heuristic
    parts = [p for p in parts if not (len(p) > 50 and re.search(r'\(\d+[rRsS],', p))]
    parts = [re.sub(r'[\s\-]+', '_', p) for p in parts]
    return parts


def clean_dataframe(df):
    df.columns = df.columns.str.strip()

    df['Drug Name'] = (df['Drug Name']
        .str.strip()
        .str.replace(r'\[\d+\]', '', regex=True)
        .str.replace(r'[\s\-/]+', '_', regex=True)
        .str.lower()
        .str.strip()
    )

    for col in ['Trade Names', 'Other Names']:
        df[col] = df[col].str.strip().apply(_clean_name_list)

    return df

def execute_load_query(cannonical_name, conn, other_names: list = None):
    cursor = conn.execute("INSERT OR IGNORE INTO drugs (cannonical_name) VALUES (?)", (cannonical_name,))
    drug_id = cursor.lastrowid or conn.execute(
        "SELECT id FROM drugs WHERE cannonical_name = ?", (cannonical_name,)
    ).fetchone()[0]

    if other_names:
        for other in other_names:
            conn.execute("INSERT OR IGNORE INTO synonyms (drug_id, other_names) VALUES (?, ?)", (drug_id, other))


def load_drugs_to_db(df, conn):
    df['combined']= [[e for l in x for e in l] for _,x in df.filter(['Trade Names','Other Names']).iterrows()]
    for _, row in df.iterrows():
        execute_load_query(cannonical_name=row['Drug Name'], conn=conn, other_names=row['combined'])
    conn.commit()


def run():
    df = pd.read_csv("llacie/database/files/raw_wiki.txt",sep='|')
    cleaned_df = clean_dataframe(df)
    ## print(cleaned_df)
    ## cleaned_df.to_csv('llacie/database/files/cleaned_wiki.txt',sep='|',index=False) ## Used for debugging. 
    conn = create_db()
    load_drugs_to_db(cleaned_df, conn)

if __name__ == '__main__':
    #sc.load_data_from_wikipedia(sc.read_vocab()) # Scrapes wikipedia and saves it to raw_wiki.txt
    run()