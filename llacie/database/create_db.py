
import pandas as pd
import sqlite3 as sql3

def create_db()->"conn":
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
    brand_names TEXT UNIQUE
    );
    """
    conn = sql3.connect("llacie/database/drugs.db")
   # conn.set_trace_callback(print)
    c = conn.cursor()
    c.execute(create_drugs_table)
    c.execute(create_synonyms_table)
    return conn

def load_xlsx_to_txt(file_path)->pd.DataFrame:
    """ Only use once if you need to moved from an excel sheet """
    df = pd.read_excel(file_path,sheet_name="Brand Names")
    df = df.drop(['Unnamed: 0','Unnamed: 3'],axis=1)
    return df

def clean_dataframe(df):
    df['Drug Name'] = df['Drug Name'].str.replace('/', '_', regex=False).str.strip()
    df['Brand Names'] = df['Brand Names'].str.replace(r',?\s*other(\[\d+\])?', '', regex=True).str.strip()
    df['Brand Names'] = df['Brand Names'].str.split(',')
    df['Brand Names'] = df['Brand Names'].apply(lambda x: [y.strip()for y in x])
    return df

def execute_load_query(cannonical_name, conn, brand_names: list = None):
    cursor = conn.execute("INSERT OR IGNORE INTO drugs (cannonical_name) VALUES (?)", (cannonical_name,))
    drug_id = cursor.lastrowid or conn.execute(
        "SELECT id FROM drugs WHERE cannonical_name = ?", (cannonical_name,)
    ).fetchone()[0]

    if brand_names:
        for brand_name in brand_names:
            conn.execute("INSERT OR IGNORE INTO synonyms (drug_id, brand_names) VALUES (?, ?)", (drug_id, brand_name))


def load_drugs_to_db(df, conn):
    for _, row in df.iterrows():
        execute_load_query(cannonical_name=row['Drug Name'], conn=conn, brand_names=row['Brand Names'])
    conn.commit()

def main():
    df = load_xlsx_to_txt("llacie/database/use_this_data/cleaned.xlsx")
    cleaned_df = clean_dataframe(df)
    conn = create_db()
    load_drugs_to_db(cleaned_df, conn)

if __name__ == '__main__':
    main()
