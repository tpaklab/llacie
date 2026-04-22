import sqlite3
import create_db as cdb
def insert_drug_basic(conn, drug_id, name):
    conn.execute("INSERT OR IGNORE INTO drugs (drug_id, cannonical_name) VALUES (?, ?)", (drug_id, name))
    
def insert_brand(conn, drug_id, brand_name):
    conn.execute("INSERT INTO brands (drug_id, brand_name) VALUES (?, ?)", (drug_id, brand_name))



def create_query(connection=None):
    if connection is None:
        raise sqlite3.DatabaseError
    while True:
        print("Please enter your query below")
        query = input()
        cursor =  connection.execute(query)
        print('Contents')
        for row in cursor:
            print(row)
        connection.commit()
        print('Query inputted.')


def intialize_drugbank():
    """Intializes tables and returns the connection"""
    conn = cdb.create_connection("llacie/database/drugbank.db")
    cdb.create_tables(conn)
    return conn 

def query_drugbank(conn):
    """ Sends query to memory database for now."""
    create_query(conn)

if __name__ == '__main__':
    conn = sqlite3.connect("llacie/database/drugbank.db")
    curose = query_drugbank(conn)

   