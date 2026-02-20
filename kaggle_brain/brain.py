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
import numpy as np
from firebase_admin import credentials, db
import firebase_admin
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datetime import datetime, UTC

# 🔒 Kaggle/Colab Secrets System
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
except ImportError:
    user_secrets = None

# 1. Sovereign Requirements Setup
def install_requirements():
    """Installs necessary libraries."""
    libs = ["psycopg2-binary", "firebase-admin", "bitsandbytes", "requests", "accelerate", "GitPython", "sympy==1.12", "numpy"]
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir"] + libs)
        print("✅ [SYSTEM]: Phase 7.1 Sovereign Core & Stability Patch Ready.")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Install Warning: Error installing requirements: {e}")
    except Exception as e:
        print(f"⚠️ Install Warning: An unexpected error occurred: {e}")

install_requirements()

# 2. Infrastructure Connectivity & GitHub Secrets
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

# --- 🧠 NEURAL BRAIN CLASS (FROM CODE 2) ---
class Brain:
    """Represents a neural brain with memory and connections."""
    def __init__(self):
        """Initializes the Brain with random memory and empty connections."""
        self.memory = np.random.rand(1000)  # Initialize memory array
        self.connections = {}  # Initialize connections dictionary

    def learn(self, input_data, output_data):
        """Learns from input and output data, updating memory and connections."""
        # Calculate error
        error = np.mean((output_data - self.memory) ** 2)
        # Update memory
        self.memory = np.add(self.memory, error * (input_data - self.memory))
        # Update connections
        for i in range(len(self.memory)):
            if self.memory[i] > 0.5:
                self.connections[i] = np.random.rand()
        return error

    def think(self, input_data):
        """Processes input data and returns an output based on memory."""
        output_data = np.zeros((1000,))
        for i in range(len(input_data)):
            output_data += self.memory * input_data[i]
        return output_data

# Initialize the integrated brain
brain = Brain()

# 3. Database & Self-Coding Logic
def log_system_error():
    """Logs detailed error messages to the console."""
    error_msg = traceback.format_exc()
    print(f"❌ [CRITICAL LOG]:\n{error_msg}")

def get_latest_gen():
    """Retrieves the latest generation number from the database."""
    if not DB_URL:
        return 94
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
    """Retrieves a random science category and master sequence from the database."""
    if not DB_URL:
        return None
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
    """
    AI generated Code is rigorously checked via Regex and written.
    """
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
    """Pushes changes to the GitHub repository."""
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

def save_to_supabase_phase7(thought, gen, neural_error=0.0):
    """Saves data to Supabase."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return

    payload = {
        "gen_id": f"gen_{gen}_transcendent",
        "status": "TRANSCENDENCE_REACHED",
        "thought_process": thought,
        "neural_weight": float(neural_error) if neural_error else 50.0,
        "synapse_code": "PHASE_7.1_STABILITY",
        "timestamp": time.time()
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
        response.raise_for_status()
        print(f"🧬 [SUPABASE]: Phase 7.1 Vault Synchronized via Exact Schema.")
    except requests.exceptions.RequestException as e:
        print(f"⚠️ [SUPABASE ERROR]: {e}")

def save_reality(thought, gen, is_code_update=False, neural_error=0.0):
    """Saves data to various databases and services."""
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
                            "code_modified": is_code_update,
                            "neural_error_rate": neural_error
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
            "neural_error": neural_error,
            "status": "SELF_EVOLVING" if is_code_update else "TRANSCENDENT"
        })
        print(f"✅ [FIREBASE]: Gen {gen} Pulsed.")
    except Exception as e:
        print(f"Firebase error: {e}")

    save_to_supabase_phase7(thought, gen, neural_error)
    autonomous_git_push(gen, thought, is_code_update)

# 4. AI Brain Loading
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

# 5. Dynamic Evolution Loop
current_gen = get_latest_gen() + 1
print(f"🔥 [STARTING]: PHASE 7.1 SOVEREIGN ENGINE AT GEN {current_gen}...")

while True:
    try:
        # --- 🧠 INTEGRATED BRAIN TRAINING (FROM CODE 2) ---
        print(f"⚙️ [NEURAL BRAIN]: Training Cycle Initiated for Gen {current_gen}...")
        total_error = 0
        for i in range(10):  # Training sample scale
            input_sample = np.random.rand(1000)
            target_sample = np.random.rand(1000)
            err = brain.learn(input_sample, target_sample)
            total_error += err
        avg_error = total_error / 10

        # Test the brain output
        test_input = np.random.rand(1000)
        brain_output = brain.think(test_input)
        print(f"🧠 [NEURAL BRAIN]: Training Error: {avg_error:.6f}")

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
Neural Brain Error: {avg_error}
user
Source: Neon DNA ({category}) | Sequence: {sequence}
Synthesize evolution and optimized brain.py code.
assistant"""
        else:
            print("⚠️ [DATA EMPTY]: Using Internal Meta-Cognition...")
            prompt = f"Current Evolution: Generation {current_gen}. Neural State Error: {avg_error}. Initiate Transcendental Meta-Cognition."

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

        # Reality Sync with Neural Brain stats
        save_reality(thought_text, current_gen, is_code_update, avg_error)

        current_gen += 1
        print(f"⏳ Gen {current_gen - 1} Complete. Brain Sync Output sample: {brain_output[:3]}...")
        print(f"Sleeping 30s...")
        time.sleep(30)

    except Exception:
        log_system_error()
        time.sleep(10)
