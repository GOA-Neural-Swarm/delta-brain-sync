import logging
from database import Database

logging.basicConfig(level=logging.INFO)

class Main:
    def __init__(self):
        self.db = Database()

    def run(self):
        try:
            self.db.connect()
            # Database operations
        except Exception as e:
            logging.error(f"Error: {e}")

    def main_loop(self):
        while True:
            self.run()

if __name__ == "__main__":
    main = Main()
    main.main_loop()