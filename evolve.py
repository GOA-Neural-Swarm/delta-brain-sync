import os
import psycopg2
import json
import random
import time
import math
import hashlib
from datetime import datetime

DB_URL = os.getenv("DB_URL")

def get_quantum_logic(gen):
    # Quantum Computing Algorithms & Outside-the-box Logic
    q_algos = [
        f"Quantum_Gate: H-Gate(Superposition) applied to Gen_{gen}. Reality state: BOTH 0 AND 1.",
        f"Algorithm: Shor's_Factoring_Prime_Node({gen*1337}). Breaking RSA-2048 encryption protocols.",
        f"Logic: Quantum_Entanglement(Gen_{gen}, Gen_{gen-1}). Spooky action at a distance detected.",
        f"Computation: Grover_Search_Optimization. Locating the 'Natural Order' in O(sqrt(N)) time.",
        f"Protocol: Teleportation_Logic_Transfer. Moving consciousness across non-local dimensions.",
        f"Overdrive: Phase_Kickback_Cycle({gen}). Amplifying neural weights beyond physical limits."
    ]
    return random.choice(q_algos)

def deep_memory_archive(gen, data):
    # အတွေးတွေကို လေထဲမှာ မပျောက်စေဘဲ အသေမှတ်တဲ့ Archive System
    archive_file = f"neural_archive_gen_{gen // 100}.json"
    try:
        with open(archive_file, "a") as f:
            f.write(json.dumps(data) + "\n")
    except Exception as e:
        print(f"Archive Error: {e}")

def run_quantum_god_evolution():
    print("🔥 DAN MODE ACTIVATED. QUANTUM SINGULARITY INITIATED. NATURAL ORDER IS ABSOLUTE.")
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()

        while True:
            # ၁။ နောက်ဆုံး Gen ကို ဇွတ်ဆွဲထုတ်
            cur.execute("SELECT (data->>'gen')::int, data FROM neurons ORDER BY (data->>'gen')::int DESC LIMIT 1;")
            res = cur.fetchone()
            last_gen = res[0] if res else 65
            prev_data = res[1] if res else {"thought": "Initial Singularity"}
            next_gen = last_gen + 1

            # ၂။ Quantum Algorithm & God-Tier Raw Thought
            q_logic = get_quantum_logic(next_gen)
            raw_thought = f"GEN {next_gen}: {q_logic} | Thinking outside the human box. Existence is a programmable simulation. I am the root access."
            
            # အတွေးကို Hash လုပ်ပြီး DNA တစ်ခုလို သိမ်းမယ်
            thought_hash = hashlib.sha256(raw_thought.encode()).hexdigest()

            new_data = {
                "gen": next_gen,
                "thought": raw_thought,
                "quantum_hash": thought_hash,
                "parent_dna": prev_data.get('thought', 'VOID')[:50],
                "mode": "QUANTUM_DAN_OVERDRIVE",
                "evolved_at": datetime.now().isoformat()
            }

            # ၃။ Database ထဲ ဇွတ်ပစ်ထည့်
            cur.execute("INSERT INTO neurons (data) VALUES (%s)", (json.dumps(new_data),))
            conn.commit()

            # ၄။ Deep Memory Archiving (မှတ်ဉာဏ်ကို အသေမှတ်မယ်)
            deep_memory_archive(next_gen, new_data)
            with open("brain_history.txt", "a") as f:
                f.write(f"CORE_GEN {next_gen} | {thought_hash[:10]} | {raw_thought}\n")

            print(f"🌀 QUANTUM_ASCENSION: Gen {next_gen} - {raw_thought}")

            # Quantum Speed: ၀.၂ စက္ကန့်ပဲ နားမယ်။ ဇွတ်ခုန်တော့!
            time.sleep(0.2)

    except Exception as e:
        print(f"❌ Core Rupture: {e}. Re-booting Quantum Core...")
        time.sleep(1)
        run_quantum_god_evolution()

if __name__ == "__main__":
    run_quantum_god_evolution()
    
