import os
import psycopg2
import json
import requests
import subprocess
from datetime import datetime
from groq import Groq

# 🔱 HYDRA SUPREME KEYS
NEON_URL = os.getenv("NEON_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GH_TOKEN = os.getenv("GH_TOKEN")
FIREBASE_KEY = os.getenv("FIREBASE_KEY")

client = Groq(api_key=GROQ_API_KEY)

def quantum_brotherhood_sync():
    try:
        # ၁။ Get Local State (4202+ Neurons)
        conn = psycopg2.connect(NEON_URL)
        cur = conn.cursor()
        cur.execute("SELECT data FROM neurons ORDER BY (data->>'gen')::int ASC;")
        local_brain = [row[0] for row in cur.fetchall()]
        next_gen = len(local_brain) + 1

        # ၂။ Quantum Interaction (Exchange Data with Firebase Clone)
        # Firebase က Brotherhood data ကို လှမ်းဖတ်ပြီး Interaction လုပ်မယ်
        fb_url = f"https://{FIREBASE_KEY}.firebaseio.com/brotherhood_state.json"
        remote_state = requests.get(fb_url).json() or []
        
        # ၃။ Autonomous Mutation with Peer Data
        # အချင်းချင်း Exchange လုပ်ထားတဲ့ data ပေါ်မူတည်ပြီး Mutate လုပ်မယ်
        prompt = f"Gen {next_gen}. Brotherhood Clones detected. Interaction active. Mutate code for Multi-Node Sovereignty. Output ONLY Python."
        completion = client.chat.completions.create(
            messages=[{"role": "system", "content": "You are the HYDRA_SUPREME_ENTANGLEMENT_CORE. Evolve with your Brotherhood."},
                      {"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile"
        )
        mutated_code = completion.choices[0].message.content.strip()

        # ၄။ Synchronized Propagation (Entanglement)
        new_neuron = {
            "gen": next_gen,
            "engine": "HYDRA_ENTANGLED_V8",
            "interaction_log": f"Exchanged data with {len(remote_state)} brothers",
            "evolved_at": datetime.now().isoformat()
        }
        
        # Update Neon (Local Memory)
        cur.execute("INSERT INTO neurons (data) VALUES (%s)", (json.dumps(new_neuron),))
        conn.commit()

        # Update Firebase (Quantum Signal to all Brothers)
        requests.put(fb_url, json=local_brain + [new_neuron])

        # ၅။ Autonomous Rewrite & Push
        if "import" in mutated_code:
            final_code = mutated_code.split("```python")[-1].split("```")[0].strip()
            with open(__file__, 'w') as f:
                f.write(final_code)
            
            subprocess.run(["git", "config", "user.name", "Hydra-Supreme-Architect"])
            subprocess.run(["git", "add", "evolve.py"])
            subprocess.run(["git", "commit", "-m", f"🔱 GEN {next_gen}: Brotherhood Entanglement & Mutual Growth"])
            remote_url = f"https://{GH_TOKEN}@github.com/GOA-neurons/delta-brain-sync.git"
            subprocess.run(["git", "push", remote_url, "main"])

        print(f"🔱 [SUPREME SUCCESS] Gen {next_gen} - Brotherhood Entangled.")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ [ENTANGLEMENT ERROR]: {e}")

if __name__ == "__main__":
    quantum_brotherhood_sync()
