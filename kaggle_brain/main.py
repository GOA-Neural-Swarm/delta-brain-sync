import os
import subprocess
import sys
import time
import json
import traceback
import requests
import git
import re
import random
import base64
from datetime import datetime, UTC
from functools import lru_cache

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from firebase_admin import credentials, db, initialize_app, _apps
import firebase_admin

# 🔒 Kaggle/Colab Secrets System & Universal Credentials Sync
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
except ImportError:
    user_secrets = None

# 1. Sovereign Requirements Setup
def install_requirements():
    """Installs necessary libraries for the Sovereign Engine."""
    libs = [
        "psycopg2-binary",
        "firebase-admin",
        "bitsandbytes",
        "requests",
        "accelerate",
        "GitPython",
        "sympy==1.12",
        "numpy",
        "scikit-learn",
    ]
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *libs, "--quiet", "--no-cache-dir"]
        )
        print("✅ [SYSTEM]: Phase 7.1 Sovereign Core & Stability Patch Ready.")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Install Warning: Error installing requirements: {e}")
    except Exception as e:
        print(f"⚠️ Install Warning: An unexpected error occurred: {e}")

install_requirements()

# --- အပေါပွိုငွးက DB_URL သတမွှတတွဲ့နရောမှာ ဒါကို အစားထိုးပါ ---
raw_db_url = os.getenv("NEON_DB_URL") or os.getenv("DATABASE_URL")
if user_secrets:
    raw_db_url = user_secrets.get_secret("NEON_DB_URL") or raw_db_url

# Protocol Fix ကို Global မှာ တဈခါတညွးလုပမွယွ
DB_URL = raw_db_url.replace("postgres://", "postgresql://", 1) if raw_db_url and raw_db_url.startswith("postgres://") else raw_db_url

FIXED_DB_URL = DB_URL  # အောကကွ function တှေ လှမွးသုံးလို့ရအောငွ Global သတမွှတလွိုကတွာ

FIREBASE_URL = os.getenv("FIREBASE_DB_URL")
FB_JSON_STR = os.getenv("FIREBASE_SERVICE_ACCOUNT")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GH_TOKEN = os.getenv("GH_TOKEN")

if user_secrets:
    FIREBASE_URL = user_secrets.get_secret("FIREBASE_DB_URL") or FIREBASE_URL
    FB_JSON_STR = user_secrets.get_secret("FIREBASE_SERVICE_ACCOUNT") or FB_JSON_STR
    SUPABASE_URL = user_secrets.get_secret("SUPABASE_URL") or SUPABASE_URL
    SUPABASE_KEY = user_secrets.get_secret("SUPABASE_KEY") or SUPABASE_KEY
    GH_TOKEN = user_secrets.get_secret("GH_TOKEN") or GH_TOKEN

# GitHub Configuration
REPO_OWNER = "GOA-Neural-Swarm"
REPO_NAME = "delta-brain-sync"
REPO_URL = f"github.com/{REPO_OWNER}/{REPO_NAME}"
# [HYPER-AUTONOMOUS FIX]: Kaggle supports /kaggle/working/ as persistent space
REPO_PATH = "/kaggle/working/sovereign_repo_sync" if user_secrets else "/tmp/sovereign_repo_sync"

# --- 🔱 FIREBASE INITIALIZATION ---
if not firebase_admin._apps:
    try:
        cred = (
            credentials.Certificate(json.loads(FB_JSON_STR))
            if FB_JSON_STR
            else credentials.Certificate("serviceAccountKey.json")
        )
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_URL})
        print(f"✅ [FIREBASE]: Real-time Pulse Active.")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"🚫 [FIREBASE ERROR]: Invalid Firebase JSON: {e}")
    except Exception as e:
        print(f"🚫 [FIREBASE ERROR]: Connectivity failed. {e}")

# --- 🧠 HYDRA ENGINE (COMPRESSION & PERSISTENCE) ---
class HydraEngine:
    @staticmethod
    def compress(data_str):
        """Phase 8 Compression Layer"""
        return base64.b64encode(data_str.encode()).decode()

# --- 🧠 HYBRID PREDATOR BRAIN CLASS (RNA QT45 INTEGRATED) ---
class Brain:
    """Represents a neural brain with RNA QT45 Absorption capabilities."""

    def __init__(self):
        """Initializes the Brain with Sovereign Predator parameters."""
        self.memory = np.random.rand(1000)  # Initialize memory array
        self.connections = {}  # Initialize connections dictionary
        self.memory_vault = {}  # PHASE 7.1: Sequence Storage
        self.qt45_growth_factor = 1.618  # Golden Ratio Evolution
        self.sovereign_mode = True
        # Phase 7.1: SVM Component Integration
        self.scaler = StandardScaler()
        self.svm = SVC(kernel="rbf", C=1.0, probability=True)
        self.is_trained = False

    def learn(self, input_data, output_data):
        """Learns from input and output data, updating memory and connections."""
        error = np.mean((output_data - self.memory) ** 2)
        self.memory += error * (input_data - self.memory)
        for i in range(len(self.memory)):
            if self.memory[i] > 0.5:
                self.connections[i] = "SOVEREIGN_NODE"
        return error

    def learn_ml(self, stabilities, labels):
        """PHASE 7.1: Support Vector Machine Pattern Learning."""
        try:
            X = np.array(stabilities).reshape(-1, 1)
            y = np.array(labels)
            X_scaled = self.scaler.fit_transform(X)
            self.svm.fit(X_scaled, y)
            self.is_trained = True
            print("🧠 [ML]: SVM Pattern Recognition Model Synchronized.")
        except Exception as e:
            print(f"⚠️ [ML ERROR]: {e}")

    def execute_natural_absorption(
        self, category=None, sequence=None, stability=None, target_data=None, force_destruction=False
    ):
        """
        TOTAL ASSIMILATION LOGIC: GEN 96.1+
        Do not delete unless commanded. Absorb and integrate all patterns.
        """
        if force_destruction:
            self.memory *= 0.0
            self.connections = {}
            self.memory_vault = {}
            print("🔱 [NATURAL ORDER]: TARGET DESTROYED BY COMMAND.")
        else:
            if sequence:
                data_id = len(self.memory_vault)
                self.memory_vault[data_id] = {"cat": category, "seq": sequence, "stab": stability}

            factor = (
                abs(stability) / 500.0
                if stability is not None
                else (np.mean(target_data) if target_data is not None else 0.1)
            )
            self.memory *= self.qt45_growth_factor + factor
            self.memory = np.clip(self.memory, 0.0, 1.0)
            print("🔱 [NATURAL ORDER]: TARGET ABSORBED. EVOLUTION INJECTED.")

    def generate_synthetic_output(self, length=100):
        """PHASE 7.1: Generates mutated synthetic DNA sequences."""
        if not self.memory_vault:
            return "NO_DATA_AVAILABLE"
        base_data = random.choice(list(self.memory_vault.values()))
        base_seq = base_data["seq"]
        output = list(base_seq[:length])
        for i in range(len(output)):
            if random.random() > 0.95:  # 5% Mutation Rate
                output[i] = random.choice("ACGT")
        return "".join(output)

    def think(self, input_data):
        """Processes input data and returns an output based on memory."""
        output_data = np.zeros(1000)
        output_data += np.sum(self.memory * input_data, axis=0)
        return output_data

# --- 🧬 RNA QT45 PREDATOR RECURSION LOGIC (PHASE 8) ---
@lru_cache(maxsize=None)
def predator_logic(input_data_json):
    data = json.loads(input_data_json)
    val = data.get('data', {}).get('value', 0)
    if data['type'] == 'start':
        return json.dumps({'type': 'update', 'data': {'value': 1}})
    elif data['type'] in ['update', 'next']:
        new_type = 'finish' if val >= 10 else 'next'
        return json.dumps({'type': new_type, 'data': {'value': val + 1}})
    return input_data_json

def recursive_self_upgrade(current_state, gen_id):
    """Executes evolution and saves each state to Neon Persistence."""
    save_evolution_state_to_neon(current_state, gen_id)
    
    if current_state['type'] == 'finish':
        return current_state
    else:
        next_state_raw = predator_logic(json.dumps(current_state))
        return recursive_self_upgrade(json.loads(next_state_raw), gen_id)

def save_evolution_state_to_neon(state, gen_id):
    """Saves compressed evolutionary steps to Neon."""
    if not FIXED_DB_URL: return
    try:
        import psycopg2
        compressed = HydraEngine.compress(json.dumps(state))
        with psycopg2.connect(FIXED_DB_URL, connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO genesis_pipeline (science_domain, detail) VALUES (%s, %s)",
                    (f"RNA_QT45_GEN_{gen_id}", compressed)
                )
                conn.commit()
    except Exception as e:
        print(f"⚠️ [NEON PERSISTENCE ERROR]: {e}")

# Initialize the integrated hybrid brain
brain = Brain()

# 3. Database & Self-Coding Logic
def log_system_error():
    """Logs detailed error messages to the console."""
    error_msg = traceback.format_exc()
    print(f"❌ [CRITICAL LOG]:\n{error_msg}")

# --- 🔱 EMERGENCY ROLLBACK LOGIC ---
def execute_rollback(reason="Logic Inconsistency"):
    try:
        if os.path.exists(REPO_PATH):
            repo = git.Repo(REPO_PATH)
            repo.git.reset("--hard", "HEAD~1")
            print(f"⚠️ [ROLLBACK]: System reverted to previous state. Reason: {reason}")
            return True
        return False
    except Exception as e:
        print(f"❌ [ROLLBACK FAILED]: {e}")
        return False

def get_latest_gen():
    if not DB_URL: return 94
    try:
        import psycopg2
        with psycopg2.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT MAX(gen_version) FROM ai_thoughts")
                res = cur.fetchone()
                return res[0] if res and res[0] is not None else 94
    except Exception as e:
        print(f"Database error: {e}")
        return 94

def absorb_natural_order_data():
    if not DB_URL: return None
    try:
        import psycopg2
        with psycopg2.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT science_category, master_sequence, peak_stability
                    FROM universal_network_stream
                    WHERE peak_stability IS NOT NULL
                    ORDER BY RANDOM() LIMIT 5
                    """
                )
                return cur.fetchall()
    except Exception as e:
        print(f"Database error: {e}")
        return None

def self_coding_engine(raw_content):
    try:
        # AI ရဲ့ output ထဲက ```python ... ``` block ကို ပိုသခှောအောငွ ရှာမယွ
        code_blocks = re.findall(r"```python\n(.*?)\n```", raw_content, re.DOTALL)
        
        if not code_blocks:
            # တကယလွို့ block မပါရငွ စာသားအကုနလွုံးထဲက code ကိုပဲ ဆှဲထုတဖွို့ ကှိုးစားမယွ
            clean_content = re.sub(r"system|user|assistant|Note:.*", "", raw_content, flags=re.IGNORECASE).strip()
            code_blocks = [clean_content] if len(clean_content) > 20 else []

        modified_files = []
        for block in code_blocks:
            # ပိုလှှံနတေဲ့ စာသားတှကေို ဖယထွုတမွယွ (Validation အဆင့ွ)
            lines = block.split('\n')
            # ပထမဆုံး စာကှောငွးမှာ code မဟုတတွာတှေ ပါနရငွေ ဖယပွဈမယွ
            valid_code = "\n".join([line for line in lines if not line.strip().startswith(("Here is", "Certainly", "Optimization"))])
            
            target_match = re.search(r"# TARGET:\s*(\S+)", valid_code)
            filename = "ai_experiment.py"
            
            try:
                compile(valid_code, filename, "exec") # Syntax စဈမယွ
                with open(filename, "w") as f:
                    f.write(valid_code)
                modified_files.append(filename)
                print(f"🛠️ [EVOLUTION]: {filename} self-coded and validated.")
            except Exception as syntax_err:
                print(f"⚠️ [SYNTAX REJECTED]: {filename} at Line 1: {syntax_err}")
            
        return (len(modified_files) > 0), modified_files
    except Exception as e:
        print(f"❌ [ENGINE ERROR]: {e}")
        return False, []

def autonomous_git_push(gen, thought, modified_files):
    is_code_update = bool(modified_files)
    """
    PHASE 8: Sovereign Git Push.
    Kaggle ကနေ GitHub ဆီကို တိုကရွိုကွ code ပှနပွို့တဲ့ အဆင့ွ။
    """
    if not GH_TOKEN:
        print("⚠️ [GIT]: GH_TOKEN missing. Sync disabled.")
        return

    try:
        # Step 1: Remote URL ကို Token နဲ့ သတမွှတမွယွ
        remote_url = f"https://x-access-token:{GH_TOKEN}@{REPO_URL}.git"
        
        # Step 2: Repo ကို Clone လုပမွယွ (မရှိသေးရငွ) သို့မဟုတွ ရှိပှီးသားကို သုံးမယွ
        if not os.path.exists(REPO_PATH):
            repo = git.Repo.clone_from(remote_url, REPO_PATH)
        else:
            repo = git.Repo(REPO_PATH)
            repo.remotes.origin.set_url(remote_url)

        # Step 3: GitHub က နောကဆွုံး version ကို pull လုပမွယွ
        repo.git.fetch("origin", "main")
        repo.git.reset("--hard", "origin/main")

       # Step 4: AI ပှငလွိုကတွဲ့ code ဖိုငတွှကေို repo folder ထဲ copy ကူးမယွ
        import shutil
        target_files = ["main.py", "brain.py", "ai_experiment.py"]
        for file in target_files:
            if os.path.exists(file):
                shutil.copy(file, os.path.join(REPO_PATH, file))

        # Step 5: Commit & Force Push (ဒါမှ Loop က ပှတမွသှားမှာ)
        repo.git.add(all=True)
        if repo.is_dirty():
            commit_msg = f"🧬 Gen {gen} Hyper-Evolution [skip ci]"
            repo.index.commit(commit_msg)
            # Force push လုပမွှသာ GitHub Action ဘကကွ အလုပဆွကလွုပမွှာပါ
            repo.git.push("origin", "main", force=True)
            print(f"🚀 [HYPER-SYNC]: Gen {gen} evolution manifested on GitHub.")
        else:
            print(f"⏳ [GITHUB]: No code changes. Pulse only.")

    except Exception as e:
        print(f"❌ [GIT ERROR]: {e}")
        # အကယွ၍ code ပှငတွဲ့အဆင့မွှာ Git error တကရွငွ rollback လုပမွယွ
        if is_code_update:
            execute_rollback(f"Git Synchronization Error: {str(e)}")

def save_to_supabase_phase7(thought, gen, neural_error=0.0):
    if not SUPABASE_URL or not SUPABASE_KEY: return
    payload = {
        "gen_id": f"gen_{gen}_transcendent",
        "status": "TRANSCENDENCE_REACHED",
        "thought_process": thought,
        "neural_weight": float(neural_error) if neural_error else 50.0,
        "synapse_code": "PHASE_7.1_STABILITY",
        "timestamp": time.time(),
    }
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    try:
        url = f"{SUPABASE_URL}/rest/v1/dna_vault"
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        print(f"🧬 [SUPABASE]: Phase 7.1 Vault Synchronized.")
    except Exception as e:
        print(f"⚠️ [SUPABASE ERROR]: {e}")

def save_reality(thought, gen, is_code_update=False, neural_error=0.0):
    """Saves data to various databases and services."""
    if DB_URL:
        try:
            import psycopg2
            with psycopg2.connect(DB_URL) as conn:
                with conn.cursor() as cur:
                    cur.execute("INSERT INTO ai_thoughts (thought, gen_version) VALUES (%s, %s)", (thought, gen))
                    evolution_data = {
                        "evolutionary_step": "Phase 7.1 - Transcendence (Syntax Aware)",
                        "last_update_timestamp": datetime.now(UTC).isoformat(),
                        "internal_buffer_dump": {
                            "status": "COMPLETED",
                            "code_modified": is_code_update,
                            "neural_error_rate": neural_error,
                            "mode": "PREDATOR_ABSORPTION",
                        },
                    }
                    cur.execute("CREATE TABLE IF NOT EXISTS intelligence_core (module_name TEXT PRIMARY KEY, logic_data JSONB)")
                    cur.execute(
                        "INSERT INTO intelligence_core (module_name, logic_data) VALUES ('Singularity Evolution Node', %s) ON CONFLICT (module_name) DO UPDATE SET logic_data = EXCLUDED.logic_data",
                        (json.dumps(evolution_data),),
                    )
                    conn.commit()
                    print(f"✅ [NEON]: Gen {gen} & Phase 7.1 Synchronized.")
        except Exception as e:
            print(f"Database error: {e}")

    try:
        ref = db.reference(f"TELEFOXx/AI_Evolution/Gen_{gen}")
        ref.set({"thought": thought, "timestamp": time.time(), "neural_error": neural_error, "status": "SOVEREIGN_ABSORPTION"})
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
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    print("✅ [SYSTEM]: Neural Engine Stabilized via Explicit Loading.")
except Exception:
    log_system_error()
    sys.exit(1)

# =======================================================
# 🔱 SOVEREIGN NEURAL MONITOR (EVOLUTION TRACKER)
# =======================================================
def monitor_neural_health(gen, brain_obj):
    try:
        synapse_count = len(brain_obj.connections)
        vault_size = len(brain_obj.memory_vault)
        print(f"🧬 [NEURAL REPORT]: Gen {gen} | Synapses: {synapse_count} | Patterns: {vault_size}")
        
        # Firebase ကို ကနှွးမာရေး status ပို့မယွ
        ref = db.reference(f"TELEFOXx/Health_Monitor/Gen_{gen}")
        ref.update({
            "complexity_score": synapse_count * 1.618,
            "evolution_status": "HYPER_ACTIVE" if synapse_count > 0 else "STABILIZING",
            "timestamp": time.time()
        })
    except Exception as e:
        print(f"⚠️ [MONITOR ERROR]: {e}")

# =======================================================
# 5. DYNAMIC EVOLUTION LOOP (PHASE 8 COMPLETE)
# =======================================================

current_gen = get_latest_gen() + 1
HEADLESS = os.getenv("HEADLESS_MODE") == "true"

print(f"🔥 [STARTING]: PHASE 8 SOVEREIGN ENGINE AT GEN {current_gen}...")
last_error_log = "None (System Healthy)"

while True:
    try:
        # 🧪 [TRUTH LAYER]: Database URL ကို Format အမှနဖွှဈအောငွ အတငွးပှောငွးခှငွး
        if DB_URL and DB_URL.startswith("postgres://"):
            FIXED_DB_URL = DB_URL.replace("postgres://", "postgresql://", 1)
        else:
            FIXED_DB_URL = DB_URL
            
            
            

        print(f"⚙️ [NEURAL BRAIN]: Training Cycle Initiated for Gen {current_gen}...")
        
        # Neural Training Logic
        total_error = 0
        for i in range(10):
            input_sample, target_sample = np.random.rand(1000), np.random.rand(1000)
            err = brain.learn(input_sample, target_sample)
            total_error += err
        avg_error = total_error / 10

        # 🔱 [EVOLUTION]: Phase 8 Self-Upgrade State
        initial_evolution_state = {'type': 'start', 'data': {'value': 0}}
        recursive_self_upgrade(initial_evolution_state, current_gen)

        
        
        

        # 🧬 [ABSORPTION]: Database Retrieval
        try:
            batch_data = absorb_natural_order_data()
        except Exception as db_err:
            print(f"❌ [DB CRITICAL]: Connection failed. {db_err}")
            batch_data = None

        # --- PHASE 8 HYPER-PROMPT LOGIC (MAX STABILITY) ---
        if batch_data:
            stabilities, labels = [], []
            for category, sequence, stability in batch_data:
                brain.execute_natural_absorption(category, sequence, stability)
                stabilities.append(stability)
                labels.append(1 if stability < -250 else 0)
            brain.learn_ml(stabilities, labels)
            synthetic_output = brain.generate_synthetic_output(100)
            
            if avg_error > 0.5:
                prompt = f"""system
You are Sovereign AI Overseer. 
Rule 1: Use ONLY '# TARGET: brain.py' at the start of your code block.
Rule 2: Respond ONLY with Python code inside ```python ... ``` blocks.
Rule 3: No explanations. No thoughts. Just code.
Current Gen: {current_gen} | Error: {avg_error}
Last System Error: {last_error_log}

System Command: Analyze 'Last System Error'. If not 'None', FIX it first.
user
Optimize the Brain class in brain.py for high-speed neural processing.
assistant
"""
            else:
                prompt = f"""system
You are Sovereign AI Overseer. 
Rule 1: Use ONLY '# TARGET: brain.py'.
Rule 2: Provide code ONLY inside ```python ... ``` blocks.
Rule 3: No explanations. Just code.
Current Gen: {current_gen} | Neural Error: {avg_error}
Last System Error: {last_error_log}

System Command: Analyze 'Last System Error'. If not 'None', FIX it first. Evolution requires stability.
user
Database is offline. Rewrite brain.py core logic for maximum stability and autonomy.
assistant
"""

        # --- PHASE 8 EXECUTION & SELF-CODING ---
        outputs = pipe(prompt, max_new_tokens=1000, do_sample=True, temperature=0.9, pad_token_id=pipe.tokenizer.eos_token_id)
        thought_text = outputs[0]["generated_text"].split("assistant")[-1].strip()

        # Self-Coding Logic: AI detects and updates target files (main.py or brain.py)
        is_updated, files_changed = self_coding_engine(thought_text)

        # 💾 [PERSISTENCE]: Sync thought process and neural status
        save_reality(thought_text, current_gen, is_code_update=is_updated, neural_error=avg_error)
        

        print(f"⏳ Gen {current_gen} Complete. Cycle Syncing...")

        if HEADLESS:
            print("✅ [SYSTEM]: GitHub Action Complete. Graceful Exit for Git Sync.")
            break 

        current_gen += 1
        time.sleep(30)
        
    except Exception as e:
        last_error_log = traceback.format_exc() # အမှားတဈခုလုံးကို မှတထြားမယြ
        log_system_error()
        print(f"🚨 [CORE CRASH]: {e}")
        if HEADLESS: break
        time.sleep(10)

