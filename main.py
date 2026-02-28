import logging
import sqlite3

# Set up logging
logging.basicConfig(level=logging.INFO)

# Connect to database (fallback to memory storage if database is offline)
try:
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
except sqlite3.Error:
    logging.info('Database offline. Falling back to memory storage.')
    cursor = None

# Core logic
def main():
    if cursor:
        # Perform database operations
        cursor.execute('SELECT * FROM table')
        results = cursor.fetchall()
    else:
        # Use memory storage
        results = []

    # Process results
    for result in results:
        logging.info(f'Processed result: {result}')

if __name__ == '__main__':
    main()