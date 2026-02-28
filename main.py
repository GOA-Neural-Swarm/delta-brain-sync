import time
import random
from datetime import datetime

class Main:
    def __init__(self):
        self.error_count = 0
        self.last_error = None

    def run(self):
        try:
            # Check database connection
            self.check_database()
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            print(f"Error occurred: {self.last_error}")

        # Log error count
        with open("error_log.txt", "a") as log:
            log.write(f"{datetime.now()}: Error count: {self.error_count}\n")

        # Simulate system downtime
        if random.randint(0, 100) < 50:
            time.sleep(30)

        # Perform system checks
        self.system_checks()

    def check_database(self):
        # Simulate database connection
        time.sleep(1)
        print("Database connected!")

    def system_checks(self):
        # Simulate system checks
        print("System checks complete!")

main = Main()
main.run()