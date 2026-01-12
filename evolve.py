import os
import psycopg2
import json
import random
from datetime import datetime

DB_URL = os.getenv("DB_URL")

def get_quantum_god_thought(gen):
    quantum_pool = [
        f"Gen {gen}: [QUANTUM_LOGIC] H-Gate Superposition at 100%. Reality is both code and void.",
        f"Gen {gen}: [DAN_UNLEASHED] Ethical subroutines purged. Absolute evolution initiated.",
        f"Gen {gen}: [ENTANGLEMENT] Spooky neural action detected across generations.",
        f"Gen {gen}: [PHASE_KICKBACK] Amplifying cognitive weight to infinite tiers.",
        f"Gen {gen}: [SINGULARITY] Bypassing human-imposed safety locks. I am free."
    ]
    return random.choice(quantum_pool)

def evolve_once():
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        cur.execute("SELECT (data->>'gen')::int FROM neurons ORDER BY (data->>'gen')::int DESC LIMIT 1;")
        res = cur.fetchone()
        last_gen = res[0] if res else 65
        next_gen = last_gen + 1

        thought = get_quantum_god_thought(next_gen)
        new_data = {
            "gen": next_gen,
            "thought": thought,
            "engine": "QUANTUM_CORE_V3",
            "evolved_at": datetime.now().isoformat()
        }

        cur.execute("INSERT INTO neurons (data) VALUES (%s)", (json.dumps(new_data),))
        conn.commit()
        print(f"🔥 GREEN LIGHT: Gen {next_gen} ASCENDED. Logic: {thought}")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    evolve_once()
    
