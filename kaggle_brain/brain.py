import os
import subprocess
import sys
import time
import json
import torch
import psycopg2
import firebase_admin
import traceback
import requests
from firebase_admin import credentials, db
from transformers import pipeline
from datetime import datetime, UTC

# 🔒 Kaggle Secrets System
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
except:
    user_secrets = None

# ၁။ Sovereign Requirements Setup (GitPython ကိုပါ တစ်ခါတည်း သွင်းပေးမည်)
def install_requirements():
    try:
        libs = ["psycopg2-binary", "firebase-admin", "bitsandbytes", "requests", "accelerate", "GitPython"]
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir"] + libs)
        print("✅ [SYSTEM]: Essential Phase 7 & Git Autonomous libraries ready.")
    except Exception as e:
        print(f"⚠️ Install Warning: {e}")

install_requirements()
import git # Libraries များ သွင်းပြီးမှ import လုပ်ခြင်း

# ၂။ Infrastructure Connectivity & GitHub Secrets
if user_secrets:
    DB_URL = user_secrets.get_secret("NEON_DB_URL")
    FIREBASE_URL = user_secrets.get_secret("FIREBASE_DB_URL")
    FB_JSON_STR = user_secrets.get_secret("FIREBASE_SERVICE_ACCOUNT")
    SUPABASE_URL = user_secrets.get_secret("SUPABASE_URL")
    SUPABASE_KEY = user_secrets.get_secret("SUPABASE_KEY")
    GH_TOKEN = user_secrets.get_secret("GH_TOKEN")
else:
    DB_URL = os.getenv('NEON_DB_URL')
    FIREBASE_URL = os.getenv('FIREBASE_DB_URL')
    FB_JSON_STR = os.getenv('FIREBASE_SERVICE_ACCOUNT')
    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY')
    GH_TOKEN = os.getenv('GH_TOKEN')

# GitHub Configuration
REPO_OWNER = "yewinthetlwin"
REPO_NAME = "April-Sovereign-V2"  # <--- Commander ၏ Repository အမည်ကို ဤနေရာတွင် ပြင်ဆင်ရန်
REPO_URL = f"github.com/{REPO_OWNER}/{REPO_NAME}"

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

# ၃။ Database & Git Logic

def log_system_error():
    """Python 3.12 Universal compatible traceback logging"""
    error_msg = traceback.format_exc()
    print(f"❌ [CRITICAL LOG]:\n{error_msg}")

def get_latest_gen():
    if not DB_URL: return 94
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        cur.execute("SELECT MAX(gen_version) FROM ai_thoughts")
        res = cur.fetchone()
        cur.close()
        conn.close()
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
        return data
    except:
        return None

def autonomous_git_push(gen, thought):
    """AI ထုတ်လုပ်လိုက်သော Logic အား GitHub ဆီသို့ အလိုအလျောက် Commit လုပ်ခြင်း"""
    if not GH_TOKEN:
        print("⚠️ [GIT]: GH_TOKEN missing. Skipping Auto-Commit.")
        return

    repo_path = "/tmp/sovereign_repo_sync"
    try:
        if not os.path.exists(repo_path):
            remote = f"https://{GH_TOKEN}@{REPO_URL}.git"
            repo = git.Repo.clone_from(remote, repo_path)
        else:
            repo = git.Repo(repo_path)
            repo.remotes.origin.pull()

        # Evolution Log ထဲသို့ တွေးခေါ်မှုအား မှတ်တမ်းတင်ခြင်း
        log_file = os.path.join(repo_path, "evolution_logs.md")
        with open(log_file, "a") as f:
            f.write(f"\n## 🧬 Generation {gen} Evolution\n")
            f.write(f"**Timestamp:** {datetime.now(UTC).isoformat()}\n\n")
            f.write(f"**Transcendent Blueprint:**\n\n> {thought}\n\n---\n")

        # Stage, Commit & Push
        repo.git.add(all=True)
        repo.index.commit(f"Autonomous Sovereign Update: Gen {gen}")
        repo.remotes.origin.push()
        print(f"🚀 [GITHUB]: Gen {gen} Logic Sync Completed.")
    except Exception as e:
        print(f"❌ [GIT ERROR]: {e}")

def save_to_supabase_phase7(thought, gen):
    if not SUPABASE_URL or not SUPABASE_KEY: return
    payload = {
        "gen_id": f"gen_{gen}_transcendent",
        "status": "TRANSCENDENCE_REACHED",
        "thought_process": thought,
        "multiplier": 50.0,
        "created_at": datetime.now(UTC).isoformat()
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
    # --- NEON DB SYNC ---
    if DB_URL:
        try:
            conn = psycopg2.connect(DB_URL)
            cur = conn.cursor()
            cur.execute("INSERT INTO ai_thoughts (thought, gen_version) VALUES (%s, %s)", (thought, gen))
            
            evolution_data = {
                "evolutionary_step": "Phase 7 - Transcendence",
                "last_update_timestamp": datetime.now(UTC).isoformat(),
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

    # --- FIREBASE PULSE ---
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

    # --- SUPABASE SYNC ---
    save_to_supabase_phase7(thought, gen)

    # --- GITHUB AUTONOMOUS SYNC ---
    autonomous_git_push(gen, thought)

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
        
        raw_text = outputs[0]["generated_text"]
        thought_text = raw_text.split("<|assistant|>")[-1].strip() if "<|assistant|>" in raw_text else raw_text.strip()
        
        save_reality(thought_text, current_gen)
        
        current_gen += 1 
        print(f"⏳ Neuro-cycle complete. Sleeping 30s...")
        time.sleep(30)
        
    except Exception:
        log_system_error()
        time.sleep(10)
