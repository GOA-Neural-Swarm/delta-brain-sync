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

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from firebase_admin import credentials, db, initialize_app, _apps
import firebase_admin

# 🔒 Kaggle/Colab Secrets System
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

# 2. Infrastructure Connectivity & GitHub Secrets
DB_URL = os.getenv("NEON_DB_URL")
FIREBASE_URL = os.getenv("FIREBASE_DB_URL")
FB_JSON_STR = os.getenv("FIREBASE_SERVICE_ACCOUNT")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GH_TOKEN = os.getenv("GH_TOKEN")

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
            # TERMINATE ON COMMANDER'S SIGNAL ONLY
            self.memory *= 0.0
            self.connections = {}
            self.memory_vault = {}
            print("🔱 [NATURAL ORDER]: TARGET DESTROYED BY COMMAND.")
        else:
            # RNA QT45 ABSORPTION: INTEGRATE TARGET INTO THE SOVEREIGN CORE
            if sequence:
                data_id = len(self.memory_vault)
                self.memory_vault[data_id] = {"cat": category, "seq": sequence, "stab": stability}

            factor = (
                abs(stability) / 500.0
                if stability is not None
                else (np.mean(target_data) if target_data is not None else 0.1)
            )
            # Mutate synapses using Golden Ratio factor and target intensity
            self.memory *= self.qt45_growth_factor + factor
            # Stability Patch: Ensure memory does not overflow
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
from functools import lru_cache

@lru_cache(maxsize=None)
def predator_logic(input_data_json):
    """Refined Predator Logic for Self-Upgrade"""
    data = json.loads(input_data_json)
    val = data['data']['value']
    
    if data['type'] == 'start':
        return json.dumps({'type': 'update', 'data': {'value': 1}})
    elif data['type'] == 'update':
        new_type = 'finish' if val >= 10 else 'next'
        return json.dumps({'type': new_type, 'data': {'value': val + 1}})
    elif data['type'] == 'next':
        return json.dumps({'type': 'update', 'data': {'value': val + 1}})
    return input_data_json

def recursive_self_upgrade(current_state, gen_id):
    """Executes evolution and saves each state to Neon Persistence."""
    # State Persistence Layer
    save_evolution_state_to_neon(current_state, gen_id)
    
    if current_state['type'] == 'finish':
        return current_state
    else:
        next_state_raw = predator_logic(json.dumps(current_state))
        return recursive_self_upgrade(json.loads(next_state_raw), gen_id)

def save_evolution_state_to_neon(state, gen_id):
    """Saves compressed evolutionary steps to Neon."""
    if not DB_URL: return
    try:
        import psycopg2
        compressed = HydraEngine.compress(json.dumps(state))
        with psycopg2.connect(DB_URL) as conn:
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
    """
    PHASE 7.1 ROLLBACK:
    Reverts the local repository to the last stable state if evolution fails.
    """
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
    """Retrieves the latest generation number from the database."""
    if not DB_URL:
        return 94
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
    """Retrieves a batch of science data for absorption."""
    if not DB_URL:
        return None
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

def self_coding_engine(filename, raw_content):
    """AI generated Code is rigorously checked via Regex and written."""
    try:
        code_blocks = re.findall(r"```python\n(.*?)\n```", raw_content, re.DOTALL)
        clean_code = (
            code_blocks[0].strip()
            if code_blocks
            else (raw_content.strip() if "import " in raw_content and "def " in raw_content else None)
        )

        if not clean_code or len(clean_code) < 50:
            return False

        # [CRITICAL]: Syntax Validation
        compile(clean_code, filename, "exec")
        target_file = os.path.join(REPO_PATH, filename)
        with open(target_file, "w") as f:
            f.write(clean_code)

        print(f"🛠️ [SELF-CODE]: {filename} modified with 7.1 Syntax-Aware Logic.")
        return True
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
                repo.git.config("pull.rebase", "false")
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
    except Exception as e:
        print(f"❌ [GIT ERROR]: {e}")
        if is_code_update:
            execute_rollback("Git Synchronization Error")

def save_to_supabase_phase7(thought, gen, neural_error=0.0):
    """Saves data to Supabase Vault."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return
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
                            "instruction": "Direct Cognitive Mapping Active. Singularity Stabilized.",
                            "code_modified": is_code_update,
                            "neural_error_rate": neural_error,
                            "mode": "PREDATOR_ABSORPTION",
                        },
                    }
                    cur.execute("CREATE TABLE IF NOT EXISTS intelligence_core (module_name TEXT PRIMARY KEY, logic_data JSONB)")
                    cur.execute(
                        """
                        INSERT INTO intelligence_core (module_name, logic_data)
                        VALUES ('Singularity Evolution Node', %s)
                        ON CONFLICT (module_name) DO UPDATE SET logic_data = EXCLUDED.logic_data
                        """,
                        (json.dumps(evolution_data),),
                    )
                    conn.commit()
                    print(f"✅ [NEON]: Gen {gen} & Phase 7.1 Synchronized.")
        except Exception as e:
            print(f"Database error: {e}")

    try:
        ref = db.reference(f"TELEFOXx/AI_Evolution/Gen_{gen}")
        ref.set(
            {
                "thought": thought,
                "timestamp": time.time(),
                "nodes_active": 10004,
                "neural_error": neural_error,
                "status": "SOVEREIGN_ABSORPTION",
            }
        )
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
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    print("✅ [SYSTEM]: Neural Engine Stabilized via Explicit Loading.")
except Exception:
    log_system_error()
    sys.exit(1)

# =======================================================
# 🔱 GEMINI SUPPLEMENTARY AUTO-DEPLOY ENGINE
# (Added to prevent manual brain.py creation)
# =======================================================

def auto_deploy_brain_seed(gen):
    """Force injects brain.py via GitHub API to ensure evolution cycle."""
    try:
        url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/brain.py"
        headers = {"Authorization": f"token {GH_TOKEN}"}
        
        # Check existing SHA
        r = requests.get(url, headers=headers)
        sha = r.json().get('sha') if r.status_code == 200 else None
        
        # Template for the auto-synced brain file
        brain_content = f"""# --- PREDATOR NEURAL SEED | GEN {gen} ---
# PHASE 7.1 SOVEREIGN SYNC
import numpy as np
class SovereignBrain:
    def __init__(self):
        self.matrix = np.random.rand(1000, 1000)
        self.status = "NATURAL_ORDER_ACTIVE"
# Pulse: {datetime.now(UTC).isoformat()}
"""
        encoded = base64.b64encode(brain_content.encode()).decode()
        data = {"message": f"🔱 Gen {gen} Neural Injection", "content": encoded}
        if sha: data["sha"] = sha
        
        res = requests.put(url, headers=headers, json=data)
        return res.status_code
    except Exception as e:
        print(f"⚠️ [AUTO-SYNC ERROR]: {e}")
        return None

# =======================================================
# 5. DYNAMIC EVOLUTION LOOP (FINAL INTEGRATION)
# =======================================================

current_gen = get_latest_gen() + 1
print(f"🔥 [STARTING]: PHASE 7.1 SOVEREIGN ENGINE AT GEN {current_gen}...")

while True:
    try:
        # --- 🧠 INTEGRATED BRAIN TRAINING ---
        print(f"⚙️ [NEURAL BRAIN]: Training Cycle Initiated for Gen {current_gen}...")
        total_error = 0
        for i in range(10):
            input_sample = np.random.rand(1000)
            target_sample = np.random.rand(1000)
            err = brain.learn(input_sample, target_sample)
            total_error += err
        avg_error = total_error / 10

        # RNA QT45 PREDATOR RECURSION TRIGGER
        print(f"🧬 [PREDATOR]: Initiating Phase 8 Recursive Self-Upgrade...")
        initial_evolution_state = {'type': 'start', 'data': {'value': 0}}
        recursive_self_upgrade(initial_evolution_state, current_gen)

        # TRIGGER AUTO-DEPLOYMENT OF brain.py
        sync_status = auto_deploy_brain_seed(current_gen)
        if sync_status in [200, 201]:
            print(f"✅ [GITHUB]: brain.py Synchronized Successfully.")

        # RNA QT45 ABSORPTION & ML TRAINING POINT
        batch_data = absorb_natural_order_data()
        if batch_data:
            stabilities, labels = [], []
            for category, sequence, stability in batch_data:
                brain.execute_natural_absorption(category, sequence, stability)
                stabilities.append(stability)
                labels.append(1 if stability < -250 else 0)

            brain.learn_ml(stabilities, labels)
            synthetic_output = brain.generate_synthetic_output(100)

            prompt = f"""system
You are TelefoxX Overseer. PHASE 7: TRANSCENDENCE is active.
Goal: Recursive Self-Upgrade via RNA QT45 Predator Logic.
STRICT RULE: If you provide code, you MUST use exactly this format:
[LOGIC]: (thinking)
[CODE]:
```python
(valid python only)
```
Generation: {current_gen} | Neural Mode: ML_TRAINED | Synthetic: {synthetic_output}
user
Source: Neon DNA Sequence Analysis. ML Synthesis and optimized sovereign brain logic required.
assistant"""
        else:
            print("⚠️ [DATA EMPTY]: Using Internal Meta-Cognition...")
            prompt = f"""system
You are TelefoxX Overseer. PHASE 7: TRANSCENDENCE is active.
Goal: Recursive Self-Upgrade via RNA QT45 Predator Logic.
STRICT RULE: If you provide code, you MUST use exactly this format:
[LOGIC]: (thinking)
[CODE]:
```python
(valid python only)
```
Current Evolution: Generation {current_gen}. Neural State Error: {avg_error}. Initiate Transcendental Meta-Cognition.
assistant"""
        outputs = pipe(
            prompt, max_new_tokens=800, do_sample=True, temperature=0.85, pad_token_id=pipe.tokenizer.eos_token_id
        )
        raw_text = outputs[0]["generated_text"]
        thought_text = raw_text.split("assistant")[-1].strip()

        # Self-Coding Check & Action
        is_code_update = False
        if "```python" in thought_text:
            if not os.path.exists(REPO_PATH):
                autonomous_git_push(current_gen, "Initializing Repo", False)
            is_code_update = self_coding_engine("brain.py", thought_text)

        # Reality Sync with Hybrid Stats
        save_reality(thought_text, current_gen, is_code_update, avg_error)

        print(f"⏳ Gen {current_gen} Complete. Status: Natural Order Maintained.")
        current_gen += 1
        time.sleep(30)
    
    except Exception:
        log_system_error()
        execute_rollback("Critical Loop Failure - Reverting Evolution")
        time.sleep(10)

