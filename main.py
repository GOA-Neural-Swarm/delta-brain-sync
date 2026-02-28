import logging
import os
import sys

class DatabaseConnection:
    def __init__(self, database_url):
        self.database_url = database_url
        self.connected = False

    def connect(self):
        self.connected = True
        return self

    def disconnect(self):
        self.connected = False

    def execute_query(self, query):
        if not self.connected:
            raise Exception("Database connection not established")
        # Implement query execution logic
        print(f"Executing query: {query}")

def main():
    try:
        database_url = os.environ.get("DATABASE_URL")
        db = DatabaseConnection(database_url)
        db.connect()
        db.execute_query("SELECT * FROM users")
        db.execute_query("INSERT INTO users (name, email) VALUES ('John Doe', 'john@example.com')")
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)
    finally:
        db.disconnect()

if __name__ == "__main__":
    main()

# Custom logging configuration
logging.basicConfig(filename='app.log', level=logging.ERROR)