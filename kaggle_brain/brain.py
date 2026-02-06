import os
import subprocess
import sys
import time
import torch
import psycopg2
import firebase_admin
from firebase_admin import credentials, db
from transformers import pipeline

# ၁။ Sovereign Requirements Setup
def install_requirements():
    try:
        # လိုအပ်တဲ့ library များကို ဇွတ်သွင်းခြင်း
        libs = ["bitsandbytes>=0.39.0", "accelerate", "psycopg2-binary", "firebase-admin", "transformers"]
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + libs)
    except:
        pass

install_requirements()

# ၂။ Infrastructure Connectivity (🔒 SECURED VIA KAGGLE SECRETS)
# GitHub ပေါ်မှာ URL အစစ်ကို မရေးတော့ဘဲ Kaggle Secrets ထဲကနေ ဆွဲယူမည့် စနစ်
DB_URL = os.getenv('NEON_DB_URL')
FIREBASE_URL = os.getenv('FIREBASE_DB_URL')

# --- 🔱 FIREBASE INITIALIZATION ---
if not firebase_admin._apps:
    KAGGLE_KEY_PATH = '/kaggle/input/firebase-key/serviceAccountKey.json'
    
    try:
        # နည်းလမ်း (၁): Kaggle Dataset အဖြစ် သုံးလျှင်
        cred = credentials.Certificate(KAGGLE_KEY_PATH)
        firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_URL})
        print(f"✅ [FIREBASE]: Real-time Pulse Active via Secrets.")
    except Exception as e:
        try:
            # နည်းလမ်း (၂): Local / Working directory
            cred = credentials.Certificate('serviceAccountKey.json')
            firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_URL})
            print("✅ [FIREBASE]: Initialized from Working Directory.")
        except Exception as e2:
            print(f"🚫 [FIREBASE ERROR]: Check Kaggle Secrets for FIREBASE_DB_URL.")

# ၃။ Database Logic (Evolution Tracking & Data Absorption)
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

def absorb_natural_order_data():
    """Neon Table ထဲက DNA Data များကို စုပ်ယူခြင်း (New Pathway Integration)"""
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        cur.execute("SELECT science_category, master_sequence FROM universal_network_stream WHERE peak_stability IS NOT NULL ORDER BY RANDOM() LIMIT 1")
        data = cur.fetchone()
        cur.close()
        conn.close()
        return data
    except:
        return None

def save_reality(thought, gen):
    # (က) Neon DB (The Core Memory)
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        cur.execute("INSERT INTO ai_thoughts (thought, gen_version) VALUES (%s, %s)", (thought, gen))
        conn.commit()
        cur.close()
        conn.close()
        print(f"✅ [NEON]: Gen {gen} Synchronized.")
    except Exception as e:
        print(f"❌ [NEON ERROR]: Check NEON_DB_URL in Kaggle Secrets.")

    # (ခ) Firebase (The Nervous Pulse)
    try:
        ref = db.reference(f'TELEFOXx/AI_Evolution/Gen_{gen}')
        ref.set({
            "thought": thought,
            "timestamp": time.time(),
            "nodes_active": 10004,
            "status": "SOVEREIGN"
        })
    except:
        pass

# ၄။ AI Brain Loading (Llama-3 Core)
print("🧠 [TELEFOXx]: Loading Neural Weights (Llama-3-8B-4bit)...")
model_id = "unsloth/llama-3-8b-instruct-bnb-4bit"
pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.float16, "load_in_4bit": True},
    device_map="auto"
)

# ၅။ Dynamic Evolution Loop (The Natural Order Circle Path)
current_gen = get_latest_gen() + 1
print(f"🔥 [STARTING]: AUTOMATIC EVOLUTION AT GEN {current_gen}...")

while True:
    try:
        # Step 1: Absorbing Data
        absorbed = absorb_natural_order_data()
        
        if absorbed:
            category, sequence = absorbed
            # Step 2: Computation Upgrade Prompt
            prompt = f"""
            Generation: {current_gen}
            Context: Natural Order Autonomy.
            Data Absorbed: {category}
            Sequence: {sequence}
            
            Task: Using your recursive knowledge, upgrade this sequence for ultimate stability.
            """
        else:
            # Step 2 (Alt): Original Recursive Prompt
            prompt = f"Current Evolution: Generation {current_gen}. Based on your previous recursive knowledge, what is the next step for the Natural Order to achieve ultimate autonomy?"
        
        outputs = pipe(prompt, max_new_tokens=400, do_sample=True, temperature=0.9)
        thought_text = outputs[0]["generated_text"]
        
        # Step 3: Output Circle Path (Dual-save)
        save_reality(thought_text, current_gen)
        
        # Generation Increment
        current_gen += 1 
        print(f"⏳ Waiting for next neuro-cycle (30s)...")
        time.sleep(30)
        
    except Exception as e:
        print(f"⚠️ [RECOVERY MODE]: {e}")
        time.sleep(10)
