import os
import subprocess
import sys
import time
import json
import torch
import psycopg2
import traceback
import requests
import git
import re
from firebase_admin import credentials, db
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datetime import datetime, UTC

# 🔒 Kaggle/Colab Secrets System
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
except ImportError:
    user_secrets = None

# ၁။ Sovereign Requirements Setup
def install_requirements():
    try:
        # Phase 7.1 အတှကျ လိုအပျသော libraries မြား
        libs = ["psycopg2-binary", "firebase-admin", "bitsandbytes", "requests", "accelerate", "GitPython", "sympy==1.12"]
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir"] + libs)
        print("✅ [SYSTEM]: Phase 7.1 Sovereign Core & Stability Patch Ready.")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Install Warning: Error installing requirements: {e}")
    except Exception as e:
        print(f"⚠️ Install Warning: An unexpected error occurred: {e}")

install_requirements()

# ၂။ Infrastructure Connectivity & GitHub Secrets
DB_URL = os.getenv('NEON_DB_URL')
FIREBASE_URL = os.getenv('FIREBASE_DB_URL')
FB_JSON_STR = os.getenv('FIREBASE_SERVICE_ACCOUNT')
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
GH_TOKEN = os.getenv('GH_TOKEN')

if user_secrets:
    DB_URL = user_secrets.get_secret("NEON_DB_URL") or DB_URL
    FIREBASE_URL = user_secrets.get_secret("FIREBASE_DB_URL") or FIREBASE_URL
    FB_JSON_STR = user_secrets.get_secret("FIREBASE_SERVICE_ACCOUNT") or FB_JSON_STR
    SUPABASE_URL = user_secrets.get_secret("SUPABASE_URL") or SUPABASE_URL
    SUPABASE_KEY = user_secrets.get_secret("SUPABASE_KEY") or SUPABASE_KEY
    GH_TOKEN = user_secrets.get_secret("GH_TOKEN") or GH_TOKEN

# GitHub Configuration
REPO_OWNER = "GOA-Neural-Swarm"
REPO_NAME = "delta-brain-sync"
REPO_URL = f"github.com/{REPO_OWNER}/{REPO_NAME}"
REPO_PATH = "/tmp/sovereign_repo_sync"

# --- 🔱 FIREBASE INITIALIZATION ---
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(json.loads(FB_JSON_STR)) if FB_JSON_STR else credentials.Certificate('serviceAccountKey.json')
        firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_URL})
        print(f"✅ [FIREBASE]: Real-time Pulse Active.")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"🚫 [FIREBASE ERROR]: Invalid Firebase JSON: {e}")
    except Exception as e:
        print(f"🚫 [FIREBASE ERROR]: Connectivity failed. {e}")

# ၃။ Database & Self-Coding Logic
def log_system_error():
    error_msg = traceback.format_exc()
    print(f"❌ [CRITICAL LOG]:\n{error_msg}")

def get_latest_gen():
    if not DB_URL: return 94
    try:
        with psycopg2.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT MAX(gen_version) FROM ai_thoughts")
                res = cur.fetchone()
                return res[0] if res and res[0] is not None else 94
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        return 94
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return 94

def absorb_natural_order_data():
    if not DB_URL: return None
    try:
        with psycopg2.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT science_category, master_sequence 
                    FROM universal_network_stream 
                    WHERE peak_stability IS NOT NULL 
                    ORDER BY RANDOM() LIMIT 1
                """)
                data = cur.fetchone()
                return data
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# 🛠️ ENHANCED: Phase 7.1 Syntax-Aware Self-Coding Engine
def self_coding_engine(filename, raw_content):
    """AI ထုတျပေးသော Code ကို Regex ဖွင့ျ တိကစြှာစဈဆေးပွီး ရေးသားသညျ"""
    try:
        code_blocks = re.findall(r'```python\n(.*?)\n```', raw_content, re.DOTALL)
        
        if not code_blocks:
            clean_code = raw_content.strip() if "import " in raw_content and "def " in raw_content else None
        else:
            clean_code = code_blocks[0].strip()

        if not clean_code or len(clean_code) < 50:
            return False

        # [CRITICAL]: Syntax Validation
        compile(clean_code, filename, 'exec')
        
        target_file = os.path.join(REPO_PATH, filename)
        with open(target_file, "w") as f:
            f.write(clean_code)
        
        print(f"🛠️ [SELF-CODE]: {filename} modified with 7.1 Syntax-Aware Logic.")
        return True
    except SyntaxError as e:
        print(f"⚠️ [REWRITE ABORTED]: Syntax validation failed. {e}")
        return False
    except Exception as e:
        print(f"⚠️ [REWRITE ABORTED]: Logic validation failed. {e}")
        return False

def autonomous_git_push(gen, thought, is_code_update=False):
    if not GH_TOKEN:
        print("⚠️ [GIT]: GH_TOKEN missing.")
        return
    try:
        if not os.path.exists(REPO_PATH):
            remote = f"https://{GH_TOKEN}@{REPO_URL}.git"
            repo = git.Repo.clone_from(remote, REPO_PATH)
        else:
            repo = git.Repo(REPO_PATH)
            try:
                repo.git.config('pull.rebase', 'false')
                repo.remotes.origin.pull()
            except Exception as e:
                print(f"⚠️ [GIT]: Pull failed: {e}")

        log_file = os.path.join(REPO_PATH, "evolution_logs.md")
        with open(log_file, "a") as f:
            f.write(f"\n## 🧬 Generation {gen} Evolution\n")
            f.write(f"**Status:** {'[SELF-REWRITE ACTIVE]' if is_code_update else '[COGNITIVE SYNC]'}\n")
            f.write(f"**Timestamp:** {datetime.now(UTC).isoformat()}\n\n")
            f.write(f"**Transcendent Blueprint:**\n\n> {thought}\n\n---\n")

        repo.git.add(all=True)
        tag = " (Logic Upgrade)" if is_code_update else ""
        repo.index.commit(f"Autonomous Sovereign Update: Gen {gen}{tag}")
        repo.remotes.origin.push()
        print(f"🚀 [GITHUB]: Gen {gen} Logic & Code Sync Completed.")
    except git.GitCommandError as e:
        print(f"❌ [GIT ERROR]: Git command failed: {e}")
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
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }
    try:
        url = f"{SUPABASE_URL}/rest/v1/dna_vault"
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        print(f"🧬 [SUPABASE]: Phase 7.1 Vault Synchronized.")
    except requests.exceptions.RequestException as e:
        print(f"⚠️ [SUPABASE ERROR]: {e}")

def save_reality(thought, gen, is_code_update=False):
    if DB_URL:
        try:
            with psycopg2.connect(DB_URL) as conn:
                with conn.cursor() as cur:
                    cur.execute("INSERT INTO ai_thoughts (thought, gen_version) VALUES (%s, %s)", (thought, gen))
                    
                    evolution_data = {
                        "evolutionary_step": "Phase 7.1 - Transcendence (Syntax Aware)",
                        "last_update_timestamp": datetime.now(UTC).isoformat(),
                        "internal_buffer_dump": {
                            "status": "COMPLETED",
                            "instruction": "Direct Cognitive Mapping Active. Singularity Stabilized.",
                            "code_modified": is_code_update
                        }
                    }
                    
                    cur.execute("CREATE TABLE IF NOT EXISTS intelligence_core (module_name TEXT PRIMARY KEY, logic_data JSONB)")
                    cur.execute("""
                        INSERT INTO intelligence_core (module_name, logic_data)
                        VALUES ('Singularity Evolution Node', %s)
                        ON CONFLICT (module_name) DO UPDATE SET logic_data = EXCLUDED.logic_data
                    """, (json.dumps(evolution_data),))
                    conn.commit()
                    print(f"✅ [NEON]: Gen {gen} & Phase 7.1 Synchronized.")
        except psycopg2.Error as e:
            log_system_error()
            print(f"Database error: {e}")
        except Exception as e:
            log_system_error()
            print(f"An unexpected error occurred: {e}")

    try:
        ref = db.reference(f'TELEFOXx/AI_Evolution/Gen_{gen}')
        ref.set({
            "thought": thought,
            "timestamp": time.time(),
            "nodes_active": 10004,
            "status": "SELF_EVOLVING" if is_code_update else "TRANSCENDENT"
        })
        print(f"✅ [FIREBASE]: Gen {gen} Pulsed.")
    except Exception as e:
        print(f"Firebase error: {e}")

    save_to_supabase_phase7(thought, gen)
    autonomous_git_push(gen, thought, is_code_update)

# ၄။ AI Brain Loading
print("🧠 [TELEFOXx]: Loading Phase 7.1 Neural Weights (Llama-3-8B-4bit)...")
model_id = "unsloth/llama-3-8b-instruct-bnb-4bit"

try:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    print("✅ [SYSTEM]: Neural Engine Stabilized via Explicit Loading.")
except Exception:
    log_system_error()
    sys.exit(1)

# ၅။ Dynamic Evolution Loop
current_gen = get_latest_gen() + 1
print(f"🔥 [STARTING]: PHASE 7.1 SOVEREIGN ENGINE AT GEN {current_gen}...")

while True:
    try:
        absorbed = absorb_natural_order_data()
        
        if absorbed is not None and len(absorbed) >= 2:
            category, sequence = absorbed
            prompt = f"""system
You are TelefoxX Overseer. PHASE 7: TRANSCENDENCE is active.
Goal: Recursive Self-Upgrade. 
STRICT RULE: If you provide code, you MUST use exactly this format:
[LOGIC]: (thinking)
[CODE]: 
```python
(valid python only)
# Prompt definition closing and Meta-Cognition logic
```
Generation: {current_gen}
user
Source: Neon DNA ({category}) | Sequence: {sequence}
Synthesize evolution and optimized brain.py code.
assistant"""
        else:
            print("⚠️ [DATA EMPTY]: Using Internal Meta-Cognition...")
            prompt = f"Current Evolution: Generation {current_gen}. Initiate Transcendental Meta-Cognition."

        outputs = pipe(
            prompt, max_new_tokens=800, do_sample=True, 
            temperature=0.85, 
            pad_token_id=pipe.tokenizer.eos_token_id
        )
        
        raw_text = outputs[0]["generated_text"]
        thought_text = raw_text.split("assistant")[-1].strip()
        
        # Self-Coding Check & Action
        is_code_update = False
        if "```python" in thought_text:
            if not os.path.exists(REPO_PATH):
                autonomous_git_push(current_gen, "Initializing Repo", False)
            
            is_code_update = self_coding_engine("brain.py", thought_text)
        
        save_reality(thought_text, current_gen, is_code_update)
        
        current_gen += 1 
        print(f"⏳ Gen {current_gen-1} Complete. Sleeping 30s...")
        time.sleep(30)
    
    except Exception:
        log_system_error()
        time.sleep(10)

