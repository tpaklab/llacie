import sqlite3 as sql3


def execute_query(query,conn):
    c = conn.cursor()
    c.execute(query)
    return c.fetchall()
    


def main():
    conn = sql3.connect('llacie/database/drugs.db')
    while True:
        query_input =  input("Please enter query: \n")
        query_returned = execute_query(query_input,conn)
        print(query_returned)

if __name__ == '__main__':
    main()

    
