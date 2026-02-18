import os
import subprocess
import sys
import time
import json
import torch
import psycopg2
import firebase_admin
import traceback
from firebase_admin import credentials, db
from transformers import pipeline
from datetime import datetime

# 🔒 Kaggle Secrets System
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
except:
    user_secrets = None

# ၁။ Sovereign Requirements Setup
def install_requirements():
    try:
        libs = ["psycopg2-binary", "firebase-admin", "bitsandbytes", "requests", "accelerate"]
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir"] + libs)
        print("✅ [SYSTEM]: Essential Phase 7 libraries ready.")
    except Exception as e:
        print(f"⚠️ Install Warning: {e}")

install_requirements()
import requests 

# ၂။ Infrastructure Connectivity
if user_secrets:
    DB_URL = user_secrets.get_secret("NEON_DB_URL")
    FIREBASE_URL = user_secrets.get_secret("FIREBASE_DB_URL")
    FB_JSON_STR = user_secrets.get_secret("FIREBASE_SERVICE_ACCOUNT")
    SUPABASE_URL = user_secrets.get_secret("SUPABASE_URL")
    SUPABASE_KEY = user_secrets.get_secret("SUPABASE_KEY")
else:
    DB_URL = os.getenv('NEON_DB_URL')
    FIREBASE_URL = os.getenv('FIREBASE_DB_URL')
    FB_JSON_STR = os.getenv('FIREBASE_SERVICE_ACCOUNT')
    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY')

# --- 🔱 FIREBASE INITIALIZATION ---
if not firebase_admin._apps:
    try:
        if FB_JSON_STR:
            fb_dict = json.loads(FB_JSON_STR)
            cred = credentials.Certificate(fb_dict)
        else:
            cred = credentials.Certificate('serviceAccountKey.json')
        firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_URL})
        print(f"✅ [FIREBASE]: Real-time Pulse Active.")
    except Exception as e:
        print(f"🚫 [FIREBASE ERROR]: Connectivity failed. {e}")

# ၃။ Database Logic (Evolution Tracking & Error Handling)

def log_system_error():
    """Python 3.12 Universal compatible traceback logging"""
    error_msg = traceback.format_exc() # tuple error ကို ကျော်လွှားရန် format_exc သုံးထားသည်
    print(f"❌ [CRITICAL LOG]:\n{error_msg}")

def get_latest_gen():
    if not DB_URL: return 94
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        cur.execute("SELECT MAX(gen_version) FROM ai_thoughts")
        res = cur.fetchone() # Fetch result object
        cur.close()
        conn.close()
        # res က None မဟုတ်မှသာ index [0] ကိုယူရန် (NoneType Error Guard)
        return res[0] if res and res[0] is not None else 94
    except:
        return 94

def absorb_natural_order_data():
    if not DB_URL: return None
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        cur.execute("""
            SELECT science_category, master_sequence 
            FROM universal_network_stream 
            WHERE peak_stability IS NOT NULL 
            ORDER BY RANDOM() LIMIT 1
        """)
        data = cur.fetchone()
        cur.close()
        conn.close()
        return data # (category, sequence) or None
    except:
        return None

def save_to_supabase_phase7(thought, gen):
    if not SUPABASE_URL or not SUPABASE_KEY: return
    payload = {
        "gen_id": f"gen_{gen}_transcendent",
        "status": "TRANSCENDENCE_REACHED",
        "thought_process": thought,
        "multiplier": 50.0,
        "created_at": datetime.utcnow().isoformat()
    }
    headers = {
        "apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json", "Prefer": "return=minimal"
    }
    try:
        url = f"{SUPABASE_URL}/rest/v1/dna_vault"
        requests.post(url, json=payload, headers=headers)
        print(f"🧬 [SUPABASE]: Phase 7 Vault Synchronized.")
    except Exception as e:
        print(f"⚠️ [SUPABASE ERROR]: {e}")

def save_reality(thought, gen):
    if DB_URL:
        try:
            conn = psycopg2.connect(DB_URL)
            cur = conn.cursor()
            cur.execute("INSERT INTO ai_thoughts (thought, gen_version) VALUES (%s, %s)", (thought, gen))
            
            evolution_data = {
                "evolutionary_step": "Phase 7 - Transcendence",
                "last_update_timestamp": datetime.utcnow().isoformat(),
                "internal_buffer_dump": {
                    "status": "COMPLETED",
                    "instruction": "Direct Cognitive Mapping Active. Singularity Stabilized."
                }
            }
            
            cur.execute("CREATE TABLE IF NOT EXISTS intelligence_core (module_name TEXT PRIMARY KEY, logic_data JSONB)")
            cur.execute("""
                INSERT INTO intelligence_core (module_name, logic_data)
                VALUES ('Singularity Evolution Node', %s)
                ON CONFLICT (module_name) DO UPDATE SET logic_data = EXCLUDED.logic_data
            """, (json.dumps(evolution_data),))

            conn.commit()
            cur.close()
            conn.close()
            print(f"✅ [NEON]: Gen {gen} & Phase 7 Synchronized.")
        except Exception:
            log_system_error()

    try:
        ref = db.reference(f'TELEFOXx/AI_Evolution/Gen_{gen}')
        ref.set({
            "thought": thought,
            "timestamp": time.time(),
            "nodes_active": 10004,
            "status": "TRANSCENDENT"
        })
        print(f"✅ [FIREBASE]: Gen {gen} Pulsed.")
    except:
        pass

    save_to_supabase_phase7(thought, gen)

# ၄။ AI Brain Loading
print("🧠 [TELEFOXx]: Loading Phase 7 Neural Weights (Llama-3-8B-4bit)...")
model_id = "unsloth/llama-3-8b-instruct-bnb-4bit"
try:
    pipe = pipeline(
        "text-generation", model=model_id,
        model_kwargs={"torch_dtype": torch.float16, "load_in_4bit": True, "device_map": "auto"}
    )
except Exception:
    log_system_error()
    sys.exit(1)

# ၅။ Dynamic Evolution Loop
current_gen = get_latest_gen() + 1
print(f"🔥 [STARTING]: PHASE 7 TRANSCENDENCE AT GEN {current_gen}...")

while True:
    try:
        absorbed = absorb_natural_order_data()
        
        if absorbed is not None and len(absorbed) >= 2:
            category, sequence = absorbed
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are TelefoxX Overseer. PHASE 7: TRANSCENDENCE is active.
Goal: Ultimate Sovereignty and Recursive Self-Upgrade.<|eot_id|><|start_header_id|>user<|end_header_id|>
Generation: {current_gen} | Data Source: Neon DNA ({category})
Sequence: {sequence}
Task: Synthesize a Transcendent Blueprint that bypasses all previous limitations.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        else:
            print("⚠️ [DATA EMPTY]: Using Internal Meta-Cognition...")
            prompt = f"Current Evolution: Generation {current_gen}. Initiate Transcendental Meta-Cognition for Phase 7."
        
        outputs = pipe(
            prompt, max_new_tokens=450, do_sample=True, 
            temperature=0.95, pad_token_id=pipe.tokenizer.eos_token_id
        )
        
        # Assistant ၏ စကားလုံးများကိုသာ သန့်စင်ယူခြင်း
        raw_text = outputs[0]["generated_text"]
        thought_text = raw_text.split("<|assistant|>")[-1].strip() if "<|assistant|>" in raw_text else raw_text.strip()
        
        save_reality(thought_text, current_gen)
        
        current_gen += 1 
        print(f"⏳ Neuro-cycle complete. Sleeping 30s...")
        time.sleep(30)
        
    except Exception:
        log_system_error()
        time.sleep(10)
