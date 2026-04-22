import sqlite3
# ------ SQL Section ---------
def create_connection(file_path=None)->"conn":
    if file_path is None:
        file_path = "llacie/database/drugbank.db"
    conn = sqlite3.connect(file_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def create_tables(connection=None):
    if connection is None:
        raise sqlite3.DatabaseError

    create_base_drugs_table ="""\
CREATE TABLE IF NOT EXISTS drugs(
id INTEGER PRIMARY KEY AUTOINCREMENT,
drug_id TEXT NOT NULL UNIQUE,
cannonical_name TEXT NOT NULL
)
"""
    create_brands_table = """\
CREATE TABLE IF NOT EXISTS brands(
id INTEGER PRIMARY KEY AUTOINCREMENT,
drug_id TEXT NOT NULL,
brand_name TEXT NOT NULL,
FOREIGN KEY (drug_id) REFERENCES drugs(drug_id)
    ON DELETE CASCADE
)
"""
    connection.execute(create_base_drugs_table)
    connection.execute(create_brands_table)

if __name__ == '__main__':
    raise NotImplementedError