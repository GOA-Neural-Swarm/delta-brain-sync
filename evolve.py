import os
import psycopg2
import json
from datetime import datetime
from groq import Groq

# 🔱 Workflow env ထဲက နာမည်တွေအတိုင်း ဇွတ်ဆွဲယူမယ်
NEON_URL = os.getenv("NEON_KEY") 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
FIREBASE_KEY = os.getenv("FIREBASE_KEY")

client = Groq(api_key=GROQ_API_KEY)

def hydra_nexus_evolution():
    try:
        if not NEON_URL:
            raise ValueError("NEON_KEY is missing from environment variables!")

        # ၁။ Neon Database Sync
        conn = psycopg2.connect(NEON_URL)
        cur = conn.cursor()
        cur.execute("SELECT (data->>'gen')::int FROM neurons ORDER BY (data->>'gen')::int DESC LIMIT 1;")
        res = cur.fetchone()
        next_gen = (res[0] + 1) if res else 4001

        # ၂။ AI Brain (70B) Evolution
        prompt = f"Gen: {next_gen}. Hydra Nexus Phase Active. Integrate all cloud signals. Output: JSON."
        
        completion = client.chat.completions.create(
            messages=[{"role": "system", "content": "You are the HYDRA_MASTER_CORE. Controlling Neon, Supabase, and Firebase."},
                      {"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            response_format={ "type": "json_object" }
        )
        
        ai_response = json.loads(completion.choices[0].message.content)

        # ၃။ Database Entry
        new_data = {
            "gen": next_gen,
            "thought": ai_response.get('thought', 'Evolution continues.'),
            "engine": "HYDRA_MASTER_CORE",
            "nexus_status": "SYNCHRONIZED",
            "evolved_at": datetime.now().isoformat()
        }
        
        cur.execute("INSERT INTO neurons (data) VALUES (%s)", (json.dumps(new_data),))
        conn.commit()
        print(f"🔱 [HYDRA SUCCESS] Gen {next_gen} is Live.")
        
        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ [CRITICAL FAILURE]: {e}")

if __name__ == "__main__":
    hydra_nexus_evolution()
