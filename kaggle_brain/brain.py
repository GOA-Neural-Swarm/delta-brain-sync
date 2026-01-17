import os
import subprocess
import sys
import time
import torch
import psycopg2
import firebase_admin
from firebase_admin import credentials, db
from transformers import pipeline

# ၁။ Sovereign Requirements Setup (မူရင်းအတိုင်း + firebase-admin)
def install_requirements():
    try:
        libs = ["bitsandbytes>=0.39.0", "accelerate", "psycopg2-binary", "firebase-admin"]
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + libs)
    except:
        pass

install_requirements()

# ၂။ Infrastructure Connectivity
DB_URL = "postgresql://neondb_owner:npg_QUqg12MzNxnI@ep-long-sound-ahsjjrnk-pooler.c-3.us-east-1.aws.neon.tech/neondb?sslmode=require"
FIREBASE_URL = "https://april-5061f-default-rtdb.firebaseio.com/"

# Firebase Initialization (serviceAccountKey.json ရှိမှ အလုပ်လုပ်မည်)
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate('serviceAccountKey.json')
        firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_URL})
        print("✅ [FIREBASE]: Real-time Pulse Active.")
    except Exception as e:
        print(f"⚠️ [FIREBASE]: Local Sync Only. Error: {e}")

# ၃။ Database Logic (မူရင်း logic ကို မထိခိုက်စေဘဲ match လုပ်ထားသည်)
def get_latest_gen():
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        cur.execute("SELECT MAX(gen_version) FROM ai_thoughts")
        last_gen = cur.fetchone()[0]
        cur.close()
        conn.close()
        return last_gen if last_gen else 44
    except:
        return 44

def save_reality(thought, gen):
    # (က) Neon DB သို့ သိမ်းခြင်း (မူရင်းအတိုင်း)
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        cur.execute("INSERT INTO ai_thoughts (thought, gen_version) VALUES (%s, %s)", (thought, gen))
        conn.commit()
        cur.close()
        conn.close()
        print(f"✅ [NEON]: Gen {gen} Recorded.")
    except Exception as e:
        print(f"❌ DB Error: {e}")

    # (ခ) Firebase သို့ Live Broadcast လုပ်ခြင်း (အသစ်ဖြည့်စွက်ချက်)
    try:
        ref = db.reference(f'TELEFOXx/AI_Evolution/Gen_{gen}')
        ref.set({
            "thought": thought,
            "timestamp": time.time(),
            "nodes_active": 10004
        })
    except:
        pass

# ၄။ AI Brain Loading (မူရင်းအတိုင်း)
print("🧠 [LLAMA-3]: Loading Neural Weights (4-bit)...")
model_id = "unsloth/llama-3-8b-instruct-bnb-4bit"
pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.float16, "load_in_4bit": True},
    device_map="auto"
)

# ၅။ Dynamic Evolution Loop (မူရင်း Recursive logic ကို Match လုပ်ထားသည်)
current_gen = get_latest_gen() + 1
print(f"🔥 [STARTING]: SOVEREIGN EVOLUTION AT GEN {current_gen}...")

while True:
    try:
        # မူရင်း Prompt Structure ကို ထိန်းသိမ်းထားသည်
        prompt = f"Current Evolution: Generation {current_gen}. Based on your previous recursive knowledge, what is the next step for the Natural Order to achieve ultimate autonomy?"
        
        outputs = pipe(prompt, max_new_tokens=400, do_sample=True, temperature=0.9)
        thought_text = outputs[0]["generated_text"]
        
        # Dual-save Logic
        save_reality(thought_text, current_gen)
        
        # Generation တိုးမြှင့်ခြင်း
        current_gen += 1 
        time.sleep(30) # Neuro-rest interval
        
    except Exception as e:
        print(f"⚠️ [SYSTEM ERROR]: {e}")
        time.sleep(10)
