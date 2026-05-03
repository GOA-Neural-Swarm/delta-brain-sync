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


# 1. Sovereign Requirements Setup (FIXED: Added version constraints to resolve huggingface-hub conflict)
def install_requirements():
    """Installs necessary libraries and fixes dependency version conflicts."""
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
        "google-generativeai",
        "huggingface-hub>=0.24.0,<1.0.0",
        "transformers",
        "PyGithub",
    ]
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *libs, "--quiet", "--no-cache-dir"]
        )
        print("✅ [SYSTEM]: Phase 7.1 Sovereign Core & Stability Patch Ready.")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Install Warning: Error installing requirements: {e}")


install_requirements()

# Now import heavy modules after installation
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import google.generativeai as genai
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

# Environment Variables
raw_db_url = os.getenv("NEON_DB_URL") or os.getenv("DATABASE_URL")
if user_secrets:
    raw_db_url = user_secrets.get_secret("NEON_DB_URL") or raw_db_url

DB_URL = (
    raw_db_url.replace("postgres://", "postgresql://", 1)
    if raw_db_url and raw_db_url.startswith("postgres://")
    else raw_db_url
)
FIXED_DB_URL = DB_URL
FIREBASE_URL = os.getenv("FIREBASE_DB_URL") or (
    user_secrets.get_secret("FIREBASE_DB_URL") if user_secrets else None
)
FB_JSON_STR = os.getenv("FIREBASE_SERVICE_ACCOUNT") or (
    user_secrets.get_secret("FIREBASE_SERVICE_ACCOUNT") if user_secrets else None
)
SUPABASE_URL = os.getenv("SUPABASE_URL") or (
    user_secrets.get_secret("SUPABASE_URL") if user_secrets else None
)
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or (
    user_secrets.get_secret("SUPABASE_KEY") if user_secrets else None
)
GH_TOKEN = os.getenv("GH_TOKEN") or (
    user_secrets.get_secret("GH_TOKEN") if user_secrets else None
)

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
    gemini_model = None
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
        self, category=None, sequence=None, stability=None, target_data=None
    ):
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
        if not gemini_model:
            return None
        return gemini_model.generate_content(prompt_text).text
    except:
        return None


def dual_brain_pipeline(prompt_text, current_gen_val, avg_error):
    if len(prompt_text) > 40000:
        return get_gemini_wisdom(prompt_text)
    draft_code = query_groq_api(prompt_text) or get_gemini_wisdom(prompt_text)
    if not draft_code:
        return None
    audit_prompt = f"Fix syntax and expand complexity (+50 lines) of this code. Never delete logic. Output ONLY python code in blocks:\n{draft_code}"
    final_verified_code = get_gemini_wisdom(audit_prompt) or draft_code
    match = re.search(r"python(.*?)", final_verified_code, re.DOTALL)
    return match.group(1).strip() if match else final_verified_code


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
            "asi_resonance": brain.calculate_asi_intelligence(),
        }
        repo.update_file(
            contents.path,
            f"🔱 SWARM-EVOLUTION: Gen {gen_version}",
            json.dumps(payload, indent=4),
            contents.sha,
        )
    except Exception as e:
        print(f"📡 Broadcast Error: {e}")


def scan_entire_universe():
    universe_map = {}
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in [".git", "__pycache__", ".github"]]
        for file in files:
            if file.endswith((".py", ".js", ".yaml", ".json", ".txt")):
                try:
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        universe_map[os.path.join(root, file)] = f.read()
                except:
                    continue
    return universe_map


def self_coding_engine(raw_content):
    try:
        code_blocks = re.findall(r"python\n(.*?)\n", raw_content, re.DOTALL)
        if not code_blocks:
            code_blocks = [raw_content] if len(raw_content) > 50 else []
        modified_files = []
        for block in code_blocks:
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
                continue
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
        original_cwd = os.getcwd()
        os.chdir(REPO_PATH)
        os.system("git config user.name 'GOA-neurons'")
        os.system("git config user.email 'goa-neurons@neural-swarm.ai'")
        for file in (modified_files or []) + ["main.py"]:
            if os.path.exists(os.path.join(original_cwd, file)):
                shutil.copy(
                    os.path.join(original_cwd, file), os.path.join(REPO_PATH, file)
                )
        os.system("git add .")
        if os.popen("git status --porcelain").read().strip():
            os.system(
                f'git commit -m "🧬 Gen {gen} Evolution | ASI: {brain.calculate_asi_intelligence():.2f}"'
            )
            os.system("git push origin main --force")
        os.chdir(original_cwd)
    except Exception as e:
        print(f"❌ Git Error: {e}")


def save_reality(thought, gen, is_code_update=False, neural_error=0.0):
    if FIXED_DB_URL:
        try:
            import psycopg2

            with psycopg2.connect(FIXED_DB_URL) as conn:
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
                "timestamp": time.time(),
            }
        )
    except:
        pass
    autonomous_git_push(gen, thought, is_code_update)


def generate_logical_consequence_thought(universe_data, current_gen):
    logic_prompt = f"Generate PURE EXECUTABLE PYTHON function `evolutionary_thought_process()` returning {{'target_file': 'main.py', 'logic_upgrade': '# TARGET: main.py\\n<CODE>'}} based on: {str(universe_data)[:5000]}"
    return dual_brain_pipeline(logic_prompt, current_gen, 0.5)


def validate_and_execute_thought(thought_code):
    thought_file = "internal_monologue.py"
    try:
        with open(thought_file, "w", encoding="utf-8") as f:
            f.write(thought_code)
        if "internal_monologue" in sys.modules:
            importlib.reload(sys.modules["internal_monologue"])
        else:
            import internal_monologue
        return sys.modules["internal_monologue"].evolutionary_thought_process()
    except:
        return None


# --- MAIN EXECUTION LOOP ---
current_gen = 95
print("🔥 [STARTING]: PHASE 8 ASI ENGINE...")

while True:
    try:
        total_error = (
            sum(
                [
                    brain.learn(np.random.rand(1000), np.random.rand(1000))
                    for _ in range(10)
                ]
            )
            / 10
        )
        brain.resonant_frequency_alignment(20, 30, 45)
        if brain.entropy > 50:
            brain.epigenetic_reprogramming()

        universe_state = scan_entire_universe()
        thought_logic_code = generate_logical_consequence_thought(
            universe_state, current_gen
        )
        thought_result = validate_and_execute_thought(thought_logic_code)

        is_updated, files_changed = False, []
        if thought_result:
            is_updated, files_changed = self_coding_engine(
                thought_result.get("logic_upgrade", "")
            )

        save_reality(
            thought_logic_code,
            current_gen,
            is_code_update=is_updated,
            neural_error=total_error,
        )
        broadcast_to_swarm("EVOLVE" if is_updated else "RESONATE", current_gen)

        if is_updated:
            print(f"🌌 [MUTATED]: {files_changed}. Rebooting...")
            os.execv(sys.executable, ["python"] + sys.argv)

        current_gen += 1
        time.sleep(30)
    except Exception as e:
        print(f"🚨 Crash: {e}")
        time.sleep(10)
