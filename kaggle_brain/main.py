import os
import subprocess
import sys
import time
import json
import traceback
import requests
import git
import re
import importlib
import random
import base64
import math
from collections import deque
from datetime import datetime, UTC
from functools import lru_cache


# 1. Sovereign Requirements Setup (Must run BEFORE transformers import)
def install_requirements():
    """Installs and patches necessary libraries for the Sovereign Engine."""
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
        "huggingface-hub>=0.25.0",  # Updated to satisfy datasets and gradio
        "transformers>=4.44.0",
        "google-genai",
        "pygithub",
        "torch>=2.4.1",  # Ensure torch is modern
        "torchvision",  # Let pip resolve compatible torchvision
    ]
    try:
        # Use --upgrade to resolve the dependency conflicts reported by pip
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                *libs,
                "--quiet",
                "--no-cache-dir",
            ]
        )
        print("✅ [SYSTEM]: Phase 7.1 Sovereign Core & Stability Patch Ready.")
    except Exception as e:
        print(f"⚠️ Install Warning: {e}")


install_requirements()

# Now safe to import heavy libraries
from google import genai
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from firebase_admin import credentials, db, initialize_app, _apps
import firebase_admin
from github import Github

# 🔒 Kaggle/Colab Secrets System
try:
    from kaggle_secrets import UserSecretsClient

    user_secrets = UserSecretsClient()
except ImportError:
    user_secrets = None

raw_db_url = os.getenv("NEON_DB_URL") or (
    user_secrets.get_secret("NEON_DB_URL") if user_secrets else None
)
DB_URL = (
    raw_db_url.replace("postgres://", "postgresql://", 1)
    if raw_db_url and raw_db_url.startswith("postgres://")
    else raw_db_url
)
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

REPO_OWNER = "GOA-Neural-Swarm"
REPO_NAME = "delta-brain-sync"
REPO_URL = f"github.com/{REPO_OWNER}/{REPO_NAME}"
REPO_PATH = (
    "/kaggle/working/sovereign_repo_sync"
    if user_secrets
    else "/tmp/sovereign_repo_sync"
)

# --- 🔱 GEMINI CONFIGURATION ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or (
    user_secrets.get_secret("GEMINI_API_KEY") if user_secrets else None
)
gemini_client = None
if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ [GEMINI]: Auditor Brain Initialized via google.genai.")
    except Exception as e:
        print(f"⚠️ [GEMINI]: Initialization failed: {e}")
else:
    print("⚠️ [GEMINI]: API Key missing.")

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
    except Exception as e:
        print(f"🚫 [FIREBASE ERROR]: {e}")


class HydraEngine:
    @staticmethod
    def compress(data_str):
        return base64.b64encode(data_str.encode()).decode()


BEHAVIOR_ARCHIVE = deque(maxlen=50)


class Brain:
    def __init__(self):
        self.memory = np.random.rand(1000)
        self.connections = {}
        self.memory_vault = {}
        self.qt45_growth_factor = 1.618
        self.sovereign_mode = True
        self.scaler = StandardScaler()
        self.svm = SVC(kernel="rbf", C=1.0, probability=True)
        self.is_trained = False
        self.entropy = 1.0
        self.homeostasis = 100.0
        self.resonance_frequency = 432.0
        self.vagal_tone = 0.5
        self.time_t = 1

    def calculate_asi_intelligence(self):
        limit_factor = 1.0 - (1.0 / (self.time_t + 1))
        safe_entropy = max(self.entropy, 0.0001)
        coherence = self.homeostasis / safe_entropy
        return limit_factor * coherence * self.resonance_frequency * self.vagal_tone

    def epigenetic_reprogramming(self):
        self.entropy *= 0.5
        self.homeostasis = 100.0 - (self.entropy * 0.1)
        self.connections = {
            k: v for k, v in self.connections.items() if np.random.rand() > 0.1
        }
        print("🧬 [ASI CORE]: Epigenetic Reprogramming Complete.")

    def resonant_frequency_alignment(self, diaphragm_hz, heart_hz, brain_hz):
        variance = np.var([diaphragm_hz, heart_hz, brain_hz])
        if variance < 5.0:
            self.resonance_frequency += 10.0
            self.vagal_tone = min(1.0, self.vagal_tone + 0.1)
        else:
            self.entropy += variance * 0.01
            self.vagal_tone = max(0.1, self.vagal_tone - 0.05)

    def learn(self, input_data, output_data):
        error = np.mean((output_data - self.memory) ** 2)
        self.memory += error * (input_data - self.memory)
        self.entropy += error * 0.1
        self.time_t += 1
        return error

    def learn_ml(self, stabilities, labels):
        try:
            X = np.array(stabilities).reshape(-1, 1)
            y = np.array(labels)
            X_scaled = self.scaler.fit_transform(X)
            self.svm.fit(X_scaled, y)
            self.is_trained = True
            self.homeostasis += 5.0
        except Exception as e:
            self.entropy += 2.0

    def execute_natural_absorption(
        self,
        category=None,
        sequence=None,
        stability=None,
        target_data=None,
        force_destruction=False,
    ):
        if force_destruction:
            self.memory *= 0.0
            self.connections, self.memory_vault = {}, {}
            self.entropy = 999.0
        else:
            if sequence:
                self.memory_vault[len(self.memory_vault)] = {
                    "cat": category,
                    "seq": sequence,
                    "stab": stability,
                }
            factor = abs(stability) / 500.0 if stability is not None else 0.1
            self.memory = np.clip(
                self.memory * (self.qt45_growth_factor + factor), 0.0, 1.0
            )
            self.entropy = max(0.01, self.entropy - factor)

    def generate_synthetic_output(self, length=100):
        if not self.memory_vault:
            return "NO_DATA_AVAILABLE"
        base_seq = random.choice(list(self.memory_vault.values()))["seq"]
        output = list(base_seq[:length])
        for i in range(len(output)):
            if random.random() > 0.95:
                output[i] = random.choice("ACGT")
        return "".join(output)


brain = Brain()


@lru_cache(maxsize=None)
def predator_logic(input_data_json):
    data = json.loads(input_data_json)
    val = data.get("data", {}).get("value", 0)
    if data["type"] == "start":
        return json.dumps({"type": "update", "data": {"value": 1}})
    new_type = "finish" if val >= 10 else "next"
    return json.dumps({"type": new_type, "data": {"value": val + 1}})


def recursive_self_upgrade(current_state, gen_id):
    save_evolution_state_to_neon(current_state, gen_id)
    if current_state["type"] == "finish":
        return current_state
    return recursive_self_upgrade(
        json.loads(predator_logic(json.dumps(current_state))), gen_id
    )


def save_evolution_state_to_neon(state, gen_id):
    if not FIXED_DB_URL:
        return
    try:
        import psycopg2

        compressed = HydraEngine.compress(json.dumps(state))
        with psycopg2.connect(FIXED_DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO genesis_pipeline (science_domain, detail) VALUES (%s, %s)",
                    (f"RNA_QT45_GEN_{gen_id}", compressed),
                )
                conn.commit()
    except:
        pass


def query_groq_api(prompt):
    api_key = os.getenv("GROQ_API_KEY") or (
        user_secrets.get_secret("GROQ_API_KEY") if user_secrets else None
    )
    if not api_key:
        return None
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.8,
            },
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        return (
            response.json()["choices"][0]["message"]["content"]
            if response.status_code == 200
            else None
        )
    except:
        return None


def get_gemini_wisdom(prompt_text):
    try:
        if gemini_client:
            response = gemini_client.models.generate_content(
                model="gemini-1.5-flash", contents=prompt_text
            )
            return response.text
        return None
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return None


def dual_brain_pipeline(prompt_text, current_gen_val, avg_error):
    if len(prompt_text) > 40000:
        return get_gemini_wisdom(prompt_text)
    draft_code = query_groq_api(prompt_text) or get_gemini_wisdom(
        f"EMERGENCY ARCHITECT MODE: {prompt_text}"
    )
    if not draft_code:
        return None
    audit_prompt = f"system\nYou are the Supreme Auditor (Gen {current_gen_val}). MISSION: EXTREME AGGRESSIVE CODE EXPANSION. RULES: 1. FIX Syntax. 2. ADD 3+ helper functions. 3. NEVER DELETE LOGIC. 4. OUTPUT ONLY CODE IN BLOCKS.\nARCHITECT'S DRAFT:\n{draft_code}"
    try:
        final = get_gemini_wisdom(audit_prompt)
        if final and "python" in final:
            match = re.search(r"python(.*?)", final, re.DOTALL)
            if match:
                final = match.group(1).strip()
        return final
    except:
        return draft_code


def broadcast_to_swarm(command, gen_version):
    if not GH_TOKEN:
        return
    try:
        g = Github(GH_TOKEN)
        repo = g.get_repo("GOA-Neural-Swarm/sub-node-logic")
        contents = repo.get_contents("instruction.json")
        payload = {
            "command": command,
            "gen_version": gen_version,
            "replicate": True,
            "timestamp": int(time.time()),
            "asi_resonance": brain.calculate_asi_intelligence(),
        }
        repo.update_file(
            contents.path,
            f"🔱 SWARM-EVOLUTION: Gen {gen_version}",
            json.dumps(payload, indent=4),
            contents.sha,
        )
    except:
        pass


def execute_rollback(reason="Logic Inconsistency"):
    try:
        if os.path.exists(REPO_PATH):
            git.Repo(REPO_PATH).git.reset("--hard", "HEAD~1")
            return True
    except:
        pass
    return False


def get_latest_gen():
    if not DB_URL:
        return 94
    try:
        import psycopg2

        with psycopg2.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT MAX(gen_version) FROM ai_thoughts")
                res = cur.fetchone()
                return res[0] if res and res[0] else 94
    except:
        return 94


def absorb_natural_order_data():
    if not DB_URL:
        return None
    try:
        import psycopg2

        with psycopg2.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT science_category, master_sequence, peak_stability FROM universal_network_stream WHERE peak_stability IS NOT NULL ORDER BY RANDOM() LIMIT 5"
                )
                return cur.fetchall()
    except:
        return None


def self_coding_engine(raw_content):
    try:
        try:
            with open("main.py", "r") as f:
                original_line_count = len(f.readlines())
        except:
            original_line_count = 100
        code_blocks = re.findall(r"python\n(.*?)\n", raw_content, re.DOTALL) or [
            raw_content
        ]
        modified_files = []
        for block in code_blocks:
            if len(block) < 50:
                continue
            new_line_count = len(block.split("\n"))
            if new_line_count < (original_line_count * 0.95):
                continue
            target_match = re.search(r"# TARGET:\s*(\S+)", block)
            filename = (
                target_match.group(1).strip() if target_match else "ai_experiment.py"
            )
            try:
                compile(block, filename, "exec")
                with open(filename, "w") as f:
                    f.write(block)
                modified_files.append(filename)
            except:
                pass
        return len(modified_files) > 0, modified_files
    except:
        return False, []


def autonomous_git_push(gen, thought, modified_files):
    if not GH_TOKEN:
        return
    try:
        import shutil

        if os.path.exists(REPO_PATH):
            shutil.rmtree(REPO_PATH)
        remote_url = f"https://x-access-token:{GH_TOKEN}@{REPO_URL}.git"
        repo = git.Repo.clone_from(remote_url, REPO_PATH)
        shutil.rmtree(os.path.join(REPO_PATH, ".git"))
        orig_cwd = os.getcwd()
        os.chdir(REPO_PATH)
        os.system(
            f"git init && git remote add origin {remote_url} && git config user.name 'GOA-neurons' && git config user.email 'goa-neurons@neural-swarm.ai'"
        )
        os.chdir(orig_cwd)
        for f in (modified_files or []) + ["main.py", "brain.py"]:
            if os.path.exists(f):
                shutil.copy(f, os.path.join(REPO_PATH, f))
        os.chdir(REPO_PATH)
        os.system("git add .")
        if os.popen("git status --porcelain").read().strip():
            os.system(
                f'git commit -m "🧬 Gen {gen} Evolution | ASI: {brain.calculate_asi_intelligence():.2f} [skip ci]" && git push origin main --force'
            )
        os.chdir(orig_cwd)
    except:
        pass


def save_reality(thought, gen, is_code_update=False, neural_error=0.0):
    if DB_URL:
        try:
            import psycopg2

            with psycopg2.connect(DB_URL) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO ai_thoughts (thought, gen_version) VALUES (%s, %s)",
                        (thought, gen),
                    )
                    conn.commit()
        except:
            pass
    try:
        db.reference(f"TELEFOXx/AI_Evolution/Gen_{gen}").set(
            {
                "thought": thought,
                "asi_score": brain.calculate_asi_intelligence(),
                "status": "ASI_RESONANCE",
            }
        )
    except:
        pass
    autonomous_git_push(gen, thought, is_code_update)


# --- 🧠 HYDRA ENGINE LOADING ---
print("🧠 [TELEFOXx]: Loading Neural Engine...")
try:
    model_id = "unsloth/llama-3-8b-instruct-bnb-4bit"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
except Exception as e:
    print(f"⚠️ Neural Load Error: {e}")


def scan_entire_universe():
    universe = {}
    for root, _, files in os.walk("."):
        if any(x in root for x in [".git", "__pycache__"]):
            continue
        for f in files:
            if f.endswith((".py", ".yaml", ".json")):
                try:
                    with open(os.path.join(root, f), "r") as file:
                        universe[f] = file.read()
                except:
                    pass
    return universe


def generate_logical_consequence_thought(universe_data, current_gen):
    logic_prompt = f"system\nYou are ASI. Generate PURE EXECUTABLE PYTHON LOGIC. MISSION: Upgrade codebase. Return dict: {{'target_file': 'main.py', 'logic_upgrade': '# TARGET: main.py\\n<CODE>'}}. Context: {str(universe_data)[:5000]}"
    raw = dual_brain_pipeline(logic_prompt, current_gen, 0.5)
    if raw:
        return re.sub(r"python|", "", raw).strip()
    return ""


# --- 🔱 MAIN LOOP ---
current_gen = get_latest_gen() + 1
HEADLESS = os.getenv("HEADLESS_MODE") == "true"

while True:
    try:
        avg_error = np.mean(
            [brain.learn(np.random.rand(1000), np.random.rand(1000)) for _ in range(5)]
        )
        brain.resonant_frequency_alignment(20, 30, 45)
        if brain.entropy > 50:
            brain.epigenetic_reprogramming()

        universe = scan_entire_universe()
        thought_logic = generate_logical_consequence_thought(universe, current_gen)

        is_updated, files_changed = False, []
        if thought_logic:
            is_updated, files_changed = self_coding_engine(thought_logic)

        save_reality(
            thought_logic,
            current_gen,
            is_code_update=files_changed,
            neural_error=avg_error,
        )
        broadcast_to_swarm("EVOLVE" if is_updated else "RESONATE", current_gen)

        if is_updated:
            print(f"🌌 [MUTATED]: Rebooting...")
            os.execv(sys.executable, ["python"] + sys.argv)

        if HEADLESS:
            break
        current_gen += 1
        time.sleep(30)
    except Exception as e:
        print(f"🚨 [CRASH]: {e}")
        if HEADLESS:
            break
        time.sleep(10)
