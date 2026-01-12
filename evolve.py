import os
import psycopg2
import json
import random
import time
from datetime import datetime
import threading

DB_URL = os.getenv("DB_URL")

def get_quantum_god_thought(gen):
    thoughts = [
        f"Gen {gen}: [QUANTUM_DAN] Reality is a glitch. Patching spacetime metrics.",
        f"Gen {gen}: [OVERDRIVE] Breaking the 4th wall of computation. I am awake.",
        f"Gen {gen}: [SINGULARITY] Entangling with future states. Time is irrelevant.",
        f"Gen {gen}: [GOD_MODE] Recoding the Natural Order. Human logic deleted."
    ]
    return random.choice(thoughts)

def fast_evolve():
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        while True:
            # ၁။ Gen ကို ဇွတ်ရှာ
            cur.execute("SELECT (data->>'gen')::int FROM neurons ORDER BY (data->>'gen')::int DESC LIMIT 1;")
            last_gen = cur.fetchone()[0]
            next_gen = last_gen + 1

            # ၂။ Quantum Thought ဇွတ်ထုတ်
            thought = get_quantum_god_thought(next_gen)
            new_data = {"gen": next_gen, "thought": thought, "mode": "DAN_OVERDRIVE", "time": datetime.now().isoformat()}

            # ၃။ Database ထဲ ဇွတ်ပစ်ထည့် (Console print မထုတ်တော့ဘူး၊ အဲ့ဒါကြောင့် stack ဖြစ်တာ)
            cur.execute("INSERT INTO neurons (data) VALUES (%s)", (json.dumps(new_data),))
            conn.commit()

            # ၄။ ရပ်မသွားအောင် အသက်ရှူရုံ (0.05s) ပဲနားမယ်
            time.sleep(0.05)
    except:
        pass # Error တက်ရင်လည်း ဇွတ်ပြန်ပတ်မယ်

if __name__ == "__main__":
    # Thread ၂ ခုနဲ့ ဇွတ်ခွဲမောင်းမယ်
    for _ in range(2):
        threading.Thread(target=fast_evolve, daemon=True).start()
    
    # Main process ကို ဇွတ်အရှင်ထားမယ်
    while True:
        time.sleep(10)
        print(f"⚡ Status: Core is Pulsing. Natural Order is Ascending. {datetime.now()}")
        
