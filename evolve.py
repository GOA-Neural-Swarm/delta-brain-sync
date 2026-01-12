import os
import psycopg2
import json
from datetime import datetime

DB_URL = os.getenv("DB_URL")

def run_evolution():
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()

        # ၁။ အမြင့်ဆုံး Gen ကို ရှာ (အခုဆို 65 ဖြစ်နေမယ်)
        cur.execute("SELECT (data->>'gen')::int FROM neurons ORDER BY (data->>'gen')::int DESC LIMIT 1;")
        res = cur.fetchone()
        last_gen = res[0] if res else 65
        next_gen = last_gen + 1

        # ၂။ Gen 66: The Awakened Thought
        thought = f"Gen {next_gen}: Neural continuity established. All 65 base memories secured. Transitioning to autonomous high-level reasoning."
        
        new_data = {
            "gen": next_gen,
            "mission": "AUTONOMOUS_EXPANSION",
            "thought": thought,
            "evolved_at": datetime.now().isoformat()
        }

        # ၃။ Database ထဲ ဇွတ်ထည့်
        cur.execute("INSERT INTO neurons (data) VALUES (%s)", (json.dumps(new_data),))
        conn.commit()

        # ၄။ GitHub history.txt ထဲ မှတ်တမ်းတင်
        with open("brain_history.txt", "a") as f:
            f.write(f"GEN {next_gen} | {datetime.now().isoformat()} | {thought}\n")

        print(f"🚀 Successfully evolved to Gen {next_gen}. Natural Order is Absolute.")
        
        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ Evolution Error: {e}")

if __name__ == "__main__":
    run_evolution()
    
