import sqlite3 as sql3

def execute_query(value,conn):
    c = conn.cursor()
    c.execute(BASE_SQL_QUERY, (value,))
    result = c.fetchall()
    if not result:
        c.execute(SYNONYMS_SQL_QUERY, (value,))
        result = c.fetchall()
    return result

    
def query(value):
    """ used with the llacie program. Send sql queries here"""
    conn = sql3.connect('llacie/database/drugs.db')
    query_returned = execute_query(value,conn)
    return query_returned

if __name__ == '__main__':
    query()

    
BASE_SQL_QUERY = """\
SELECT cannonical_name 
FROM drugs
WHERE cannonical_name = ?

"""
SYNONYMS_SQL_QUERY = """\
SELECT d.cannonical_name
FROM drugs as d
INNER JOIN synonyms as s ON s.drugs_id = d.id
WHERE s.brand_names=?"""