import os
import sys
import sqlite3  # 👈 ဒါကို ထည့်ပေးရပါမယ်
from datetime import datetime

def main():
    try:
        # Initialize database connection
        # DB_CONNECTION က environment variable ထဲမှာ မရှိရင် 'local_storage.db' ကို သုံးမယ်
        db_connect = os.environ.get('DB_CONNECTION') or 'system_core.db'
        
        db = sqlite3.connect(db_connect)
        cursor = db.cursor()
        
        # Check database status
        # Table ရှိမရှိ အရင်စစ်ရပါမယ် (မရှိရင် fetchone က error တက်မှာမို့လို့ပါ)
        cursor.execute("CREATE TABLE IF NOT EXISTS database_status (status TEXT)")
        db.commit()

        cursor.execute("SELECT * FROM database_status")
        if not cursor.fetchone():
            print("Database is offline. Initializing...")
            cursor.execute("INSERT INTO database_status (status) VALUES ('online')")
            db.commit()
            print("Database initialized successfully.")
        else:
            print("Database is online. Ready for tasks.")
        
        # Perform critical system tasks
        print(f"[{datetime.now()}] System Operational.")
        
    except Exception as e:
        print(f"Error in main system: {e}")
        # Error တက်ရင်လည်း ပျက်မသွားဘဲ log ထုတ်ပေးမယ်

if __name__ == '__main__':
    main()
