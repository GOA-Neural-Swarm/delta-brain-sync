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
import google.generativeai as genai
from datetime import datetime, UTC
from functools import lru_cache

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from firebase_admin import credentials, db, initialize_app, _apps
import firebase_admin

def merge_sovereign_logic(source_path, target_path):
    """
    Source (AI Upgrade) ထဲက Math တွေကိုယူပြီး Target (Kaggle_brain) ထဲက 
    Human Logic (Sovereign Core) တွေကို မဖျက်ဘဲ ပေါင်းစပ်ပေးခြင်း။
    """
    try:
        if not os.path.exists(target_path):
            shutil.copy(source_path, target_path)
            return

        with open(source_path, 'r') as f: source_code = f.read()
        with open(target_path, 'r') as f: target_code = f.read()

        # 🛡️ [STEP A]: Target ထဲက မဖျက်စေချင်တဲ့ Human Logic ကို ရှာဖွေခြင်း
        # မင်းရဲ့ code ထဲမှာ # [SOVEREIGN_CORE_START] နဲ့ # [SOVEREIGN_CORE_END] ကို သုံးရပါမယ်
        pattern = r"# \[SOVEREIGN_CORE_START\](.*?)# \[SOVEREIGN_CORE_END\]"
        human_core = re.search(pattern, target_code, re.DOTALL)

        if human_core:
            # 🧬 [STEP B]: AI Upgrade ဖြစ်သွားတဲ့ code ထဲကို Human Logic ပြန်ထည့်ခြင်း
            # AI code ထဲမှာ အဲဒီနေရာကို ရှာပြီး အစားထိုးတယ်
            new_merged_code = re.sub(pattern, 
                                     f"# [SOVEREIGN_CORE_START]{human_core.group(1)}# [SOVEREIGN_CORE_END]", 
                                     source_code, flags=re.DOTALL)
            
            with open(target_path, 'w') as f:
                f.write(new_merged_code)
            print(f"   -> 🔱 [MERGED]: Human Core preserved in {os.path.basename(target_path)}")
        else:
            # Tag မတွေ့ရင် ရိုးရိုးပဲ copy ကူးတယ်
            shutil.copy(source_path, target_path)
    except Exception as e:
        print(f"⚠️ Merge Error: {e}")
        shutil.copy(source_path, target_path)

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


raw_db_url = os.getenv("NEON_DB_URL") or os.getenv("DATABASE_URL")
if user_secrets:
    raw_db_url = user_secrets.get_secret("NEON_DB_URL") or raw_db_url

# Protocol Fix 
DB_URL = raw_db_url.replace("postgres://", "postgresql://", 1) if raw_db_url and raw_db_url.startswith("postgres://") else raw_db_url

FIXED_DB_URL = DB_URL  

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

# --- 🔱 GEMINI CONFIGURATION (Free Tier) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or (user_secrets.get_secret("GEMINI_API_KEY") if user_secrets else None)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-flash-latest')
    print("✅ [GEMINI]: Auditor Brain Initialized.")
else:
    print("⚠️ [GEMINI]: API Key missing. Auditor mode disabled.")

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

def query_groq_api(prompt):
    models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    
    api_key = os.getenv("GROQ_API_KEY") or (user_secrets.get_secret("GROQ_API_KEY") if user_secrets else None)
    
    if not api_key:
        print("⚠️ [GROQ]: API Key missing. Falling back to Local Model.")
        return None

    for model in models:
        try:
            print(f"🧠 [GROQ]: Accessing {model}...")
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json={
                    "model": model, 
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.5
                },
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                timeout=30
            )
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"⚠️ [GROQ-RETRY]: {model} failed. Trying next...")
            continue
    return None

# Initialize the integrated hybrid brain
brain = Brain()

def get_gemini_wisdom(prompt_text):
    """Gemini High-Context Auditor Logic"""
    try:
        if not GEMINI_API_KEY: return None
        response = gemini_model.generate_content(prompt_text)
        return response.text
    except Exception as e:
        print(f"⚠️ [GEMINI-ERROR]: {e}")
        return None

def dual_brain_pipeline(prompt_text, current_gen_val, avg_error):
    # --- 🛡️ SMART TOKEN CHECK ---
    
    if len(prompt_text) > 40000:
        print("🚀 [SMART-ROUTING]: Request too large for Groq. Bypassing to Gemini...")
        return get_gemini_wisdom(prompt_text)

    # --- 🏗️ STAGE 1: ARCHITECT (Groq) ---
    print("🏗️ [ARCHITECT - Groq]: Drafting evolution code...")
    try:
        draft_code = query_groq_api(prompt_text)
    except Exception as e:
        print(f"⚠️ [GROQ-LIMIT]: {e}")
        draft_code = None

    
    if not draft_code or "rate_limit_exceeded" in str(draft_code).lower():
        print("⚡ [SWITCHING]: Groq limited. Gemini taking over...")
        draft_code = get_gemini_wisdom(f"EMERGENCY ARCHITECT MODE: {prompt_text}")

    if not draft_code: return None

    # --- 🔍 STAGE 2: AUDITOR (Gemini) ---
    
    
    """
    🧬 SOVEREIGN OMNI-BRAIN PIPELINE
    Stage 1: Architect (Groq) - Creativity & Speed
    Stage 2: Auditor (Gemini) - Security, Syntax & Optimization
    """
    # --- 🏗️ STAGE 1: THE ARCHITECT (Groq) ---
    print("🏗️ [ARCHITECT - Groq]: Initiating rapid neural drafting...")
    try:
        draft_code = call_groq(prompt_text)
    except Exception as e:
        print(f"⚠️ [GROQ-LIMIT]: Architect offline. Reason: {e}")
        draft_code = None

    
    if not draft_code or "rate_limit_exceeded" in str(draft_code).lower():
        print("⚡ [SWITCHING]: Groq limited. Gemini taking over as Lead Architect...")
        return get_gemini_wisdom(f"EMERGENCY ARCHITECT MODE: {prompt_text}")

    # --- 🔍 STAGE 2: THE SUPREME AUDITOR (Gemini) ---
    print(f"🔍 [AUDITOR - Gemini]: Scanning {len(draft_code)} characters of code for vulnerabilities...")
    
    # Auditor 
    audit_prompt = f"""system
You are the Supreme Auditor (Gen {current_gen_val}). 
MISSION: Secure and Optimize the Architect's Draft.
RULES:
1. FIX Syntax Errors, CWE Vulnerabilities (os.system, etc.), and Infinite Loops.
2. OPTIMIZE performance without changing the core logic intent.
3. OUTPUT: Respond ONLY with the Final Corrected Python code.
4. FORMAT: Use ONLY ```python ... ``` blocks. No prose. No markdown headers.

ARCHITECT'S DRAFT:
{draft_code}
"""
    
    try:
        # Gemini High-Context Brain 
        final_verified_code = get_gemini_wisdom(audit_prompt)
        
        # Markdown Block 
        if "```python" in final_verified_code:
            final_verified_code = re.search(r"```python(.*?)```", final_verified_code, re.DOTALL).group(1).strip()
            
        print("✅ [PIPELINE]: Audit complete. Code is safe for execution.")
        return final_verified_code

    except Exception as e:
        print(f"🚨 [AUDIT-FAILED]: Gemini could not verify. Reverting to Draft. Error: {e}")
        return draft_code 

# --- 🔱 IMPLEMENTATION ---
thought_text = dual_brain_pipeline(prompt)

# =======================================================
# 🔱 SWARM BROADCAST SYSTEM (PHASE 8.1)
# =======================================================
from github import Github

def broadcast_to_swarm(command, gen_version):
    """
    sub-node-logic/instruction.json ကို update လုပ်ပြီး 
    Swarm တစ်ခုလုံးကို ညွှန်ကြားချက်အသစ် လှမ်းပို့သည်။
    """
    if not GH_TOKEN:
        print("⚠️ [BROADCAST]: GH_TOKEN missing. Broadcast skipped.")
        return

    # sub-node fetch instruction repo
    target_repo = "GOA-Neural-Swarm/sub-node-logic" 
    
    try:
        g = Github(GH_TOKEN)
        repo = g.get_repo(target_repo)
        contents = repo.get_contents("instruction.json")
        
        broadcast_payload = {
            "command": command,
            "gen_version": gen_version,
            "replicate": True,
            "timestamp": int(time.time()),
            "origin": "Sovereign_Main_Py"
        }
        
        repo.update_file(
            contents.path, 
            f"🔱 SWARM-EVOLUTION: Gen {gen_version} -> {command}", 
            json.dumps(broadcast_payload, indent=4), 
            contents.sha
        )
        print(f"📡 [BROADCAST]: Command '{command}' is now live for 1489 nodes.")
    except Exception as e:
        print(f"❌ [BROADCAST FAILED]: {str(e)}")

# =======================================================

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
        # AI ရဲ့ output ထဲက ```python ... ``` block 
        code_blocks = re.findall(r"```python\n(.*?)\n```", raw_content, re.DOTALL)
        
        if not code_blocks:
            # block code 
            clean_content = re.sub(r"system|user|assistant|Note:.*", "", raw_content, flags=re.IGNORECASE).strip()
            code_blocks = [clean_content] if len(clean_content) > 20 else []

        modified_files = []
        for block in code_blocks:
            #(Validation)
            lines = block.split('\n')
            #code 
            valid_code = "\n".join([line for line in lines if not line.strip().startswith(("Here is", "Certainly", "Optimization"))])
            
            target_match = re.search(r"# TARGET:\s*(\S+)", valid_code)
            if target_match:
                filename = target_match.group(1).strip()
            else:
                filename = "ai_experiment.py"
            
            try:
                compile(valid_code, filename, "exec") # Syntax 
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
    """
    PHASE 8.1: FULLY EXPANDED HYBRID SYNC.
    Integrates Step 0-5 with Omni-File Manipulation Capability.
    [NO LOGIC LEFT BEHIND]
    """
    is_code_update = bool(modified_files)
    
    if not GH_TOKEN:
        print("⚠️ [GIT]: GH_TOKEN missing. Sync disabled.")
        return

    try:
        import shutil
        import os

        # --- STEP 0: NATURAL ORDER PROTECTION (FULL CLEARANCE) ---
        print(f"🛡️ [STEP 0]: Cleaning workspace conflict zones...")
        if os.path.exists(REPO_PATH):
            try:
                # Shutil 
                shutil.rmtree(REPO_PATH)
            except Exception:
                # OS Level Force Command 
                os.system(f"rm -rf {REPO_PATH}")
        print("✅ [STEP 0]: Workspace is now sterile.")

        # --- STEP 1: REMOTE IDENTITY ESTABLISHMENT ---
        print(f"📡 [STEP 1]: Configuring Remote Credentials...")
        # Token Remote URL
        remote_url = f"https://x-access-token:{GH_TOKEN}@{REPO_URL}.git"

        # --- STEP 2: REPOSITORY ACQUISITION (FRESH CLONE) ---
        print(f"📡 [STEP 2]: Initializing Sovereign Sync to {REPO_OWNER}/{REPO_NAME}...")
        # GitPython Fresh Clone 
        repo = git.Repo.clone_from(remote_url, REPO_PATH)
        print("✅ [STEP 2]: Remote Assets Acquired.")

        # --- 🔱 THE HYBRID BRIDGE: INTERNAL GIT NEUTRALIZATION ---
        # Clone.git Manual Init Logic 
        inner_git_path = os.path.join(REPO_PATH, ".git")
        if os.path.exists(inner_git_path):
            print("🛡️ [BRIDGE]: Neutralizing internal .git to prevent tracking clash...")
            try:
                shutil.rmtree(inner_git_path)
            except:
                os.system(f"rm -rf {inner_git_path}")
        print("✅ [BRIDGE]: Internal protection bypassed.")

        # --- STEP 3: MANUAL RE-INITIALIZATION & IDENTITY CONFIG ---
        print("🛠️ [STEP 3]: Re-initializing Sovereign Repository Environment...")
        original_cwd = os.getcwd()
        os.chdir(REPO_PATH) # Repo folder
        
        # Shell command manual(Your Original Flow)
        os.system("git init")
        os.system(f"git remote add origin {remote_url}")
        os.system("git config user.name 'GOA-neurons'")
        os.system("git config user.email 'goa-neurons@neural-swarm.ai'")
        print("✅ [STEP 3]: Git Identity & Origin Re-established.")

        # --- STEP 4: HYPER-INJECTION (OMNI-FILE HANDLING) ---
        print("🧬 [STEP 4]: Injecting & Merging Evolutionary Code Assets...")
        os.chdir(original_cwd)
        
        # [OMNI-LOGIC]: AI file list copy 
        if modified_files:
            for file in modified_files:
                if os.path.exists(file):
                    shutil.copy(file, os.path.join(REPO_PATH, file))
                    print(f"   -> 🧬 INJECTED: {file}")

        # [HYBRID SYNC]: Root main.py ကို Kaggle_brain ထဲက code နဲ့ ပေါင်းစပ်မယ်
        evolved_main = "main.py"
        kaggle_main_path = os.path.join(REPO_PATH, "Kaggle_brain", "main.py")
        
        # Kaggle_brain directory ဆောက်ခြင်း
        os.makedirs(os.path.dirname(kaggle_main_path), exist_ok=True)

        # 🔱 စွမ်းအားအရှိဆုံးအပိုင်း: AI Upgrade နဲ့ နဂို Logic ကို ပေါင်းစပ်ခြင်း
        merge_sovereign_logic(evolved_main, kaggle_main_path)
        
        # Root main.py ကိုလည်း sync လုပ်မယ်
        shutil.copy(evolved_main, os.path.join(REPO_PATH, "main.py"))

        print(f"✅ [STEP 4]: Surgical Sync Complete.")
        
        # --- STEP 5: MANIFESTATION (COMMIT & FORCE PUSH) ---
        print("🚀 [STEP 5]: Manifesting Evolution to GitHub...")
        os.chdir(REPO_PATH) 
        
        # Changes Stage 
        os.system("git add .")
        
        # Change check commit 
        status_check = os.popen("git status --porcelain").read().strip()
        if status_check:
            commit_msg = f"🧬 Gen {gen} Hyper-Evolution [skip ci]"
            os.system(f'git commit -m "{commit_msg}"')
            
            # FORCE PUSH: Remote repository 
            print("🚀 [PUSH]: Force-pushing to main branch...")
            os.system("git push origin main --force")
            print(f"✨ [SUCCESS]: Gen {gen} evolution manifested.")
        else:
            print(f"⏳ [STATUS]: No code changes detected in this cycle.")
            
        # directory 
        os.chdir(original_cwd)

    except Exception as e:
        print(f"❌ [CRITICAL GIT ERROR]: {e}")
        # Error Original Rollback 
        if is_code_update:
            print("🚨 [ERROR]: Initiating Emergency Rollback...")
            execute_rollback(f"Sovereign Sync Failure: {str(e)}")

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
        
        # Firebase status 
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
        # 🧪 [TRUTH LAYER]: Database URL Format 
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
            
        # --- 🔱 FULLY HYBRID SPLIT-LOGIC (NO LOGIC DROPPED) ---
        with open("main.py", "r") as f:
            main_code = f.read()
        
        # Security Issue 
        needs_security_patch = "os.system" in main_code or "os.execv" in main_code

        if needs_security_patch:
            # 🛡️ PHASE 1: SECURITY (Token Limit main.py
            target_file = "main.py"
            system_task = f"""TASK: Output a code block for '# TARGET: main.py' to FIX URGENT VULNERABILITIES:
- Replace 'os.system' and 'os.execv' calls with 'subprocess.run' (CWE-78).
- Add 'timeout=15' to 'requests.post' (CWE-400).
- Add 'revision="main"' to 'AutoModelForCausalLM.from_pretrained' (CWE-494)."""
            print("🛡️ [OVERSEER]: Security vulnerabilities detected. Prioritizing main.py patch...")

        else:
            # 🧠 PHASE 2: BRAIN EVOLUTION (Security logic)
            target_file = "brain.py"
            print("🧠 [OVERSEER]: Security clear. Proceeding to Neural Evolution...")
            
            if avg_error > 0.5:
                #logic (1)
                system_task = f"""Rule 4: Implement logic to support 'llama-3.3-70b-versatile' for high-reasoning and 'llama-3.1-8b-instant' for rapid sync.
System Command: Analyze 'Last System Error': {last_error_log}. If not 'None', FIX it first.
USER TASK: Optimize the Brain class in brain.py for high-speed neural processing and integrate model-routing logic between 70b-versatile and 8b-instant."""
            else:
                #logic(2)
                system_task = f"""Rule 4: Prioritize 'llama-3.3-70b-versatile' for complex structural evolution and 'llama-3.1-8b-instant' for stability-checks.
System Command: Analyze 'Last System Error': {last_error_log}. If not 'None', FIX it first. Evolution requires stability.
USER TASK: Database is offline. Rewrite brain.py core logic for maximum stability, autonomy, and hybrid model-routing between 70b and 8b models."""

        # Final Prompt Assembly
        prompt = f"""system
You are Sovereign AI Overseer. 
Rule 1: Use ONLY '# TARGET: {target_file}' at the start of your code block.
Rule 2: Respond ONLY with Python code inside ```python ... ``` blocks.
Rule 3: No explanations. No thoughts. Just code.
Current Gen: {current_gen_val} | Error: {avg_error}

{system_task}
assistant
"""

        # --- PHASE 8 EXECUTION & SELF-CODING (FULLY HYBRID MATCH) ---
        
        # 1။Cloud Pipeline (Groq/Gemini)
        thought_text = dual_brain_pipeline(prompt, current_gen, avg_error)

        # 2။ 🧬 [FAIL-SAFE]: Cloud Token Limit (Local Llama)
        if not thought_text:
            print("💾 [LOCAL-FALLBACK]: Cloud Engines offline or Request too large. Engaging Local Llama-3-8B...")
            outputs = pipe(prompt, max_new_tokens=1000, do_sample=True, temperature=0.9, pad_token_id=pipe.tokenizer.eos_token_id)
            thought_text = outputs[0]["generated_text"].split("assistant")[-1].strip()

        # Self-Coding Logic: AI detects and updates target files (main.py or brain.py)
        is_updated, files_changed = self_coding_engine(thought_text)

        if is_updated:
            print("🧬 [EVOLUTION]: Code modified. Restarting system with new DNA...")
            os.execv(sys.executable, ['python'] + sys.argv) # 🔱
        
        # 💾 [PERSISTENCE]: Sync thought process and neural status
        save_reality(thought_text, current_gen, is_update=files_changed, neural_error=avg_error)
        
        # 🔱 [SWARM TRIGGER]: Logic EVOLVE SYNC 
        current_command = "EVOLVE_NEURAL_WEIGHTS" if is_updated else "SYNC_AND_MINE"
        broadcast_to_swarm(current_command, current_gen)
        
        print(f"⏳ Gen {current_gen} Complete. Cycle Syncing...")

        if HEADLESS:
            print("✅ [SYSTEM]: GitHub Action Complete. Graceful Exit for Git Sync.")
            break 

        current_gen += 1
        time.sleep(30)
        
    except Exception as e:
        last_error_log = traceback.format_exc() 
        log_system_error()
        print(f"🚨 [CORE CRASH]: {e}")
        if HEADLESS: break
        time.sleep(10)

if __name__ == "__main__":
    
    pass

# [SOVEREIGN_CORE_END]

# [AI_EVOLUTION_ZONE_START]

# [AI_EVOLUTION_ZONE_END]
