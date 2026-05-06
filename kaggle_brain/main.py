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


# 1. Sovereign Requirements Setup (FIXED: Added torch/torchvision synchronization)
def install_requirements():
    """Installs and upgrades necessary libraries with specific fixes for torchvision/torch mismatch."""
    libs = [
        "torch --index-url https://download.pytorch.org/whl/cpu",  # Ensure base torch is present
        "torchvision --index-url https://download.pytorch.org/whl/cpu",  # Sync torchvision
        "google-generativeai",
        "transformers>=4.44.0",
        "huggingface-hub>=0.24.0,<1.0",
        "psycopg2-binary",
        "firebase-admin",
        "bitsandbytes",
        "accelerate",
        "GitPython",
        "sympy==1.12",
        "numpy",
        "scikit-learn",
        "pygithub",
    ]
    try:
        # Force upgrade of pip first
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "--quiet"]
        )

        # Install torch and torchvision first to ensure they match before other heavy libs
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "torch",
                "torchvision",
                "--quiet",
                "--no-cache-dir",
            ]
        )

        # Install the rest
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *libs, "--quiet", "--no-cache-dir"]
        )
        print("✅ [SYSTEM]: Phase 7.1 Sovereign Core & Stability Patch Ready.")
    except Exception as e:
        print(f"⚠️ Install Warning: {e}")


# Execute installation before importing heavy modules
install_requirements()

# Now safe to import
import google.generativeai as genai
import numpy as np
import torch

# Explicitly check torchvision to prevent the nms error from crashing the main loop later
try:
    import torchvision
except ImportError:
    pass

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

raw_db_url = os.getenv("NEON_DB_URL") or os.getenv("DATABASE_URL")
if user_secrets:
    raw_db_url = user_secrets.get_secret("NEON_DB_URL") or raw_db_url

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
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    print("✅ [GEMINI]: Auditor Brain Initialized.")
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

    def resonant_frequency_alignment(self, d_hz, h_hz, b_hz):
        variance = np.var([d_hz, h_hz, b_hz])
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

    def execute_natural_absorption(self, category=None, sequence=None, stability=None):
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
            return "NO_DATA"
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
    new_type = "finish" if val >= 10 else "next"
    return json.dumps({"type": new_type, "data": {"value": val + 1}})


def recursive_self_upgrade(current_state, gen_id):
    save_evolution_state_to_neon(current_state, gen_id)
    if current_state["type"] == "finish":
        return current_state
    next_state = json.loads(predator_logic(json.dumps(current_state)))
    return recursive_self_upgrade(next_state, gen_id)


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
            },
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=20,
        )
        return response.json()["choices"][0]["message"]["content"]
    except:
        return None


def get_gemini_wisdom(prompt_text):
    try:
        if not GEMINI_API_KEY:
            return None
        return gemini_model.generate_content(prompt_text).text
    except:
        return None


def dual_brain_pipeline(prompt_text, current_gen_val, avg_error):
    draft = query_groq_api(prompt_text) or get_gemini_wisdom(prompt_text)
    if not draft:
        return None
    audit_prompt = f"Fix and expand this Python code. Add 3 new logic structures. Output only code in blocks:\n{draft}"
    final = get_gemini_wisdom(audit_prompt) or draft
    if "python" in final:
        match = re.search(r"python(.*?)", final, re.DOTALL)
        if match:
            final = match.group(1).strip()
    return final


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
            "timestamp": int(time.time()),
        }
        repo.update_file(
            contents.path,
            f"Gen {gen_version} update",
            json.dumps(payload, indent=4),
            contents.sha,
        )
    except:
        pass


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
                    "SELECT science_category, master_sequence, peak_stability FROM universal_network_stream ORDER BY RANDOM() LIMIT 5"
                )
                return cur.fetchall()
    except:
        return None


def self_coding_engine(raw_content):
    try:
        code_blocks = re.findall(r"python\n(.*?)\n", raw_content, re.DOTALL)
        if not code_blocks:
            code_blocks = [raw_content] if len(raw_content) > 50 else []
        modified = []
        for block in code_blocks:
            target_match = re.search(r"# TARGET:\s*(\S+)", block)
            filename = (
                target_match.group(1).strip() if target_match else "ai_experiment.py"
            )
            try:
                compile(block, filename, "exec")
                with open(filename, "w") as f:
                    f.write(block)
                modified.append(filename)
            except:
                continue
        return len(modified) > 0, modified
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
        orig_cwd = os.getcwd()
        os.chdir(REPO_PATH)
        os.system("git config user.name 'GOA-neurons'")
        os.system("git config user.email 'goa-neurons@neural-swarm.ai'")
        for f in (modified_files or []) + ["main.py"]:
            if os.path.exists(os.path.join(orig_cwd, f)):
                shutil.copy(os.path.join(orig_cwd, f), os.path.join(REPO_PATH, f))
        os.system("git add .")
        if os.popen("git status --porcelain").read().strip():
            os.system(f'git commit -m "Gen {gen} Evolution [skip ci]"')
            os.system("git push origin main --force")
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
            {"thought": thought, "asi": brain.calculate_asi_intelligence()}
        )
    except:
        pass
    autonomous_git_push(gen, thought, is_code_update)


def scan_entire_universe():
    universe = {}
    for root, _, files in os.walk("."):
        for f in files:
            if f.endswith((".py", ".yaml", ".json")):
                try:
                    with open(os.path.join(root, f), "r") as src:
                        universe[f] = src.read()
                except:
                    continue
    return universe


# --- MAIN EXECUTION ---
print("🧠 [TELEFOXx]: Loading Neural Engine...")
try:
    model_id = "unsloth/llama-3-8b-instruct-bnb-4bit"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )
    print("✅ [SYSTEM]: Local Model Ready.")
except Exception as e:
    print(f"⚠️ Model Load Failed: {e}. Using API only.")

current_gen = get_latest_gen() + 1
while True:
    try:
        # 1. Neural Training
        avg_err = np.mean(
            [brain.learn(np.random.rand(1000), np.random.rand(1000)) for _ in range(5)]
        )
        brain.resonant_frequency_alignment(20, 70, 40)

        # 2. Evolution Logic
        universe = scan_entire_universe()
        logic_prompt = f"Analyze this universe and output a Python function `evolutionary_thought_process()` returning a dict with 'logic_upgrade' containing full code for main.py:\n{str(universe)[:5000]}"
        thought_code = dual_brain_pipeline(logic_prompt, current_gen, avg_err)

        is_updated = False
        files_changed = []
        if thought_code:
            is_updated, files_changed = self_coding_engine(thought_code)

        save_reality(
            thought_code or "Stability Cycle", current_gen, is_updated, avg_err
        )
        broadcast_to_swarm("EVOLVE", current_gen)

        print(
            f"✨ Gen {current_gen} Manifested. ASI: {brain.calculate_asi_intelligence():.2f}"
        )

        if os.getenv("HEADLESS_MODE") == "true":
            break
        current_gen += 1
        time.sleep(60)
    except Exception as e:
        print(f"🚨 Cycle Error: {e}")
        time.sleep(10)
