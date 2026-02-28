import os
import sys
import sqlite3
import recovery # 👈 Recovery module ကို import လုပ်ထားတယ်
from datetime import datetime

def recovery_action():
    """Perform recovery actions in case of failure"""
    print("🚨 [CRITICAL]: Database failure detected. Initiating recovery...")
    try:
        recovery.recover_from_failure()
        print("✅ [RECOVERED]: System restored by recovery module.")
    except Exception as e:
        print(f"❌ [RECOVERY FAILED]: {e}")

def main():
    try:
        db_connect = os.environ.get('DB_CONNECTION') or 'system_core.db'
        db = sqlite3.connect(db_connect)
        cursor = db.cursor()
        
        # Database check logic
        cursor.execute("CREATE TABLE IF NOT EXISTS database_status (status TEXT)")
        cursor.execute("SELECT * FROM database_status")
        
        if not cursor.fetchone():
            cursor.execute("INSERT INTO database_status (status) VALUES ('online')")
            db.commit()
            print("Database initialized.")
        
        print(f"[{datetime.now()}] AGI Master Link Active.")
        
    except Exception as e:
        print(f"Error: {e}")
        # 🛡️ အမှားတက်တာနဲ့ recovery action ကို လှမ်းခေါ်လိုက်မယ်
        recovery_action()

if __name__ == '__main__':
    main()
