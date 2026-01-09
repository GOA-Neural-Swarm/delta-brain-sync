import os
import json
import firebase_admin
from firebase_admin import credentials, firestore
import psycopg2
import requests
import time

def main():
    try:
        print("🌀 DELTA LOOP WAKING UP...")

        # Neon Connection
        conn = psycopg2.connect(os.environ.get('NEON_DATABASE_URL'))
        cur = conn.cursor()
        
        # Supabase မသေအောင် ၅ ခုစီပဲ ဇွတ်ပို့မယ် (Test အနေနဲ့)
        cur.execute("SELECT * FROM evolution_data LIMIT 5;") 
        rows = cur.fetchall()
        print(f"🐘 Neon: Fetched {len(rows)} records.")

        # Firebase Setup
        raw_json = os.environ.get('FIREBASE_SERVICE_ACCOUNT').replace('\\n', '\n')
        service_account_info = json.loads(raw_json, strict=False)
        cred = credentials.Certificate(service_account_info)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        print("🔥 Firestore: Ready.")

        # Supabase Sync Logic
        supa_url = os.environ.get('SUPABASE_URL') + "/rest/v1/evolution_logs"
        headers = {
            "apikey": os.environ.get('SUPABASE_KEY'),
            "Authorization": f"Bearer {os.environ.get('SUPABASE_KEY')}",
            "Content-Type": "application/json"
        }

        for row in rows:
            # မင်းရဲ့ Table Structure အတိုင်း ဒီမှာ data ပြင်
            payload = {"log_data": str(row)} 
            requests.post(supa_url, headers=headers, json=payload)
            print(f"🛰️ Supabase: Synced 1 row.")
            time.sleep(1) # Rate limit protection

        print("🏁 DELTA LOOP CYCLE FINISHED!")

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
    
