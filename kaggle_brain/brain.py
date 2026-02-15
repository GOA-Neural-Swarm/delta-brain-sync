import os
import subprocess
import sys
import time
import json
import torch
import psycopg2
import firebase_admin
from firebase_admin import credentials, db
from transformers import pipeline
# 🔱 Kaggle Secrets System
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
except:
    user_secrets = None

# ၁။ Sovereign Requirements Setup
def install_requirements():
    try:
        # GPU အထောက်အပံ့အတွက် လိုအပ်သော libraries များ
        libs = ["bitsandbytes>=0.39.0", "accelerate", "psycopg2-binary", "firebase-admin", "transformers"]
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + libs)
    except Exception as e:
        print(f"⚠️ Install Warning: {e}")

install_requirements()

# ၂။ Infrastructure Connectivity (🔒 SECURED VIA KAGGLE SECRETS)
# Kaggle UI ထဲက Add-ons > Secrets ထဲမှာ ဒီ Key တွေကို အသေအချာ Add ထားရပါမယ်
if user_secrets:
    DB_URL = user_secrets.get_secret("NEON_DB_URL")
    FIREBASE_URL = user_secrets.get_secret("FIREBASE_DB_URL")
    # Firebase Service Account JSON ကို Secret တစ်ခုတည်းမှာ String အဖြစ် ထည့်ထားလျှင် ပိုကောင်းသည်
    FB_JSON_STR = user_secrets.get_secret("FIREBASE_SERVICE_ACCOUNT")
else:
    DB_URL = os.getenv('NEON_DB_URL')
    FIREBASE_URL = os.getenv('FIREBASE_DB_URL')
    FB_JSON_STR = None

# --- 🔱 FIREBASE INITIALIZATION ---
if not firebase_admin._apps:
    try:
        if FB_JSON_STR:
            # Secrets မှတဆင့် JSON ကို Load လုပ်ခြင်း
            fb_dict = json.loads(FB_JSON_STR)
            cred = credentials.Certificate(fb_dict)
        else:
            # Local File မှ Load လုပ်ခြင်း
            cred = credentials.Certificate('serviceAccountKey.json')
            
        firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_URL})
        print(f"✅ [FIREBASE]: Real-time Pulse Active.")
    except Exception as e:
        print(f"🚫 [FIREBASE ERROR]: Connectivity failed. {e}")

# ၃။ Database Logic (Evolution Tracking & Data Absorption)
def get_latest_gen():
    if not DB_URL: return 44
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
    if not DB_URL: return None
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        # မင်းရဲ့ မူလ Query အတိုင်း DNA Data ဆွဲထုတ်ခြင်း
        cur.execute("""
            SELECT science_category, master_sequence 
            FROM universal_network_stream 
            WHERE peak_stability IS NOT NULL 
            ORDER BY RANDOM() LIMIT 1
        """)
        data = cur.fetchone()
        cur.close()
        conn.close()
        return data
    except:
        return None

def save_reality(thought, gen):
    # (က) Neon DB (The Core Memory)
    if DB_URL:
        try:
            conn = psycopg2.connect(DB_URL)
            cur = conn.cursor()
            cur.execute("INSERT INTO ai_thoughts (thought, gen_version) VALUES (%s, %s)", (thought, gen))
            conn.commit()
            cur.close()
            conn.close()
            print(f"✅ [NEON]: Gen {gen} Synchronized.")
        except Exception as e:
            print(f"❌ [NEON ERROR]: {e}")

    # (ခ) Firebase (The Nervous Pulse)
    try:
        ref = db.reference(f'TELEFOXx/AI_Evolution/Gen_{gen}')
        ref.set({
            "thought": thought,
            "timestamp": time.time(),
            "nodes_active": 10004,
            "status": "SOVEREIGN"
        })
        print(f"✅ [FIREBASE]: Gen {gen} Pulsed.")
    except:
        pass

# ၄။ AI Brain Loading (Llama-3 Core - Optimized for Kaggle T4 GPU)
print("🧠 [TELEFOXx]: Loading Neural Weights (Llama-3-8B-4bit)...")
model_id = "unsloth/llama-3-8b-instruct-bnb-4bit"
try:
    pipe = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={
            "torch_dtype": torch.float16, 
            "load_in_4bit": True,
            "device_map": "auto"
        }
    )
except Exception as e:
    print(f"❌ [BRAIN LOAD ERROR]: {e}")
    sys.exit(1)

# ၅။ Dynamic Evolution Loop (The Natural Order Circle Path)
current_gen = get_latest_gen() + 1
print(f"🔥 [STARTING]: AUTOMATIC EVOLUTION AT GEN {current_gen}...")

while True:
    try:
        # Step 1: Absorbing Data (Neon DNA)
        absorbed = absorb_natural_order_data()
        
        if absorbed:
            category, sequence = absorbed
            prompt = f"""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are TelefoxX Overseer. Natural Order Autonomy is the goal.
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Generation: {current_gen}
            Data Absorbed: {category}
            Sequence: {sequence}
            
            Task: Using recursive knowledge, upgrade this sequence for ultimate stability.
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
        else:
            prompt = f"Current Evolution: Generation {current_gen}. What is the next step for the Natural Order to achieve ultimate autonomy?"
        
        # Step 2: Generation
        outputs = pipe(
            prompt, 
            max_new_tokens=400, 
            do_sample=True, 
            temperature=0.9,
            pad_token_id=pipe.tokenizer.eos_token_id
        )
        thought_text = outputs[0]["generated_text"].split("<|assistant|>")[-1].strip()
        
        # Step 3: Output Circle Path (Dual-save to Neon & Firebase)
        save_reality(thought_text, current_gen)
        
        # Generation Increment
        current_gen += 1 
        print(f"⏳ Neuro-cycle complete. Sleeping 30s...")
        time.sleep(30)
        
    except Exception as e:
        print(f"⚠️ [RECOVERY MODE]: {e}")
        time.sleep(10)
        
