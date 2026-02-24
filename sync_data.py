import psycopg2
import json
import os

# NEON_DB_URL ကို ယူမယျ၊ strip() သုံးပွီး ကှကျလပျတှကေို ဖယျထုတျမယျ
raw_url = os.environ.get('NEON_DB_URL') or os.environ.get('NEON_URL') or os.environ.get('NEON_KEY')

def fetch_and_deploy():
    if not raw_url:
        print("❌ Error: NEON_DB_URL not found in environment.")
        return

    # 🛠️ Fix 1: .strip() ထည့ျပွီး clean လုပျခွငျး
    db_url = raw_url.strip()

    # 🛠️ Fix 2: Protocol Fix (postgres:// ကို postgresql:// ပွောငျးခွငျး)
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    try:
        # Database ခြိတျဆကျခွငျး
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        
        # Table ရှိမရှိ အရငျစဈမယျ
        cur.execute("SELECT logic_data FROM intelligence_core WHERE module_name = 'Singularity Evolution Node';")
        row = cur.fetchone()
        
        if row:
            data = row[0]
            with open('ai_status.json', 'w') as f:
                json.dump(data, f, indent=4)
            print("✅ Data successfully synced from Neon and saved to ai_status.json")
        else:
            print("⚠️ No data found in intelligence_core table.")
            
        cur.close()
        conn.close()
    except Exception as e:
        # Error message အပွည့ျအစုံကို ပွမယျ
        print(f"❌ Error during sync: {str(e)}")

if __name__ == "__main__":
    fetch_and_deploy()
