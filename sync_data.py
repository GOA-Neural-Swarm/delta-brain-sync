import psycopg2
import json
import os

# GitHub Secrets ထဲမှာ သိမ်းထားမယ့် NEON_URL ကို ယူသုံးမယ်
NEON_URL = os.environ.get('NEON_URL')

def fetch_and_deploy():
    try:
        # Database ချိတ်ဆက်ခြင်း
        conn = psycopg2.connect(NEON_URL)
        cur = conn.cursor()
        
        # Phase 6 ရဲ့ နောက်ဆုံး ဒေတာကို ဆွဲထုတ်ခြင်း
        cur.execute("SELECT logic_data FROM intelligence_core WHERE module_name = 'Singularity Evolution Node';")
        data = cur.fetchone()[0]
        
        # ai_status.json ဖိုင်အဖြစ် သိမ်းဆည်းခြင်း
        with open('ai_status.json', 'w') as f:
            json.dump(data, f, indent=4)
            
        print("Data successfully synced from Neon.")
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    fetch_and_deploy()
