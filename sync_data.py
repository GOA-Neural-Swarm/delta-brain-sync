import psycopg2
import json
import os

# YAML ထဲက env: နာမည်တွေနဲ့ အတိအကျတူအောင် ယူထားတယ်
# NEON_URL (သို့) NEON_DB_URL နှစ်ခုလုံးကို စစ်ပေးထားတယ်
raw_url = os.environ.get('NEON_URL') or os.environ.get('NEON_DB_URL') or os.environ.get('NEON_KEY')

def fetch_and_deploy():
    if not raw_url:
        print("❌ Error: Neon Connection URL not found in environment.")
        return

    # 🛠️ DSN Fix: Connection string ကို သန့်ရှင်းအောင်လုပ်ခြင်း
    # ရှေ့နောက် space ဖြတ်မယ်၊ postgres:// ကို postgresql:// ပြောင်းမယ်
    db_url = raw_url.strip()
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

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
            
        cur.close()
        conn.close()
    except Exception as e:
        # Error တက်ရင် ဘယ်နေရာမှာလဲဆိုတာ သေချာသိရအောင် error message အပြည့်အစုံပြမယ်
        print(f"❌ Error during sync: {str(e)}")

if __name__ == "__main__":
    fetch_and_deploy()
