import os
import sqlite3

def recover():
    # Check for database journal file
    if os.path.exists("agi_system.db-journal"):
        # Remove journal file to recover database
        os.remove("agi_system.db-journal")
        
        # Attempt to connect to the recovered database
        try:
            conn = sqlite3.connect('agi_system.db')
            conn.close()
            print("Database recovery successful")
        except sqlite3.Error as e:
            print(f"Error recovering database: {e}")

# Call the recover function
recover()