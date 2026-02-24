import psycopg2
import json
import os

# NEON_URL ကို ယူမယ်၊ မရှိရင် NEON_KEY ကို ရှာမယ်
raw_url = os.environ.get('NEON_DB_URL') or os.environ.get('NEON_KEY')

def fetch_and_deploy():
    if not raw_url:
        print("❌ Error: NEON_DB_URL not found in environment.")
        return

    # 🛠️ Protocol Fix: postgres:// ကို postgresql:// ပြောင်းခြင်း
    db_url = raw_url.replace("postgres://", "postgresql://", 1) if raw_url.startswith("postgres://") else raw_url

    try:
        # Database ချိတ်ဆက်ခြင်း
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        
        # Table ရှိမရှိ အရင်စစ်မယ် (Safety Check)
        cur.execute("SELECT logic_data FROM intelligence_core WHERE module_name = 'Singularity Evolution Node';")
        row = cur.fetchone()
        
        if row:
            data = row[0]
            with open('ai_status.json', 'w') as f:
                json.dump(data, f, indent=4)
            print("✅ Data successfully synced from Neon and saved to ai_status.json")
        else:
            print("⚠️ No data found in intelligence_core table.")
            
        conn.close()
    except Exception as e:
        print(f"❌ Error during sync: {e}")

if __name__ == "__main__":
    fetch_and_deploy()
