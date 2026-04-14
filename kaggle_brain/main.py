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
from firebase_admin import credentials, db, initialize_app, _apps
import firebase_admin

# 🔒 Kaggle/Colab Secrets System
try:
    from kaggle_secrets import UserSecretsClient

    user_secrets = UserSecretsClient()
except ImportError:
    user_secrets = None


def install_requirements():
    """Installs necessary libraries and fixes version conflicts."""
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
        "transformers --upgrade",
        "huggingface-hub>=0.24.0",
    ]
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *libs, "--quiet", "--no-cache-dir"]
        )
        print("✅ [SYSTEM]: Phase 7.1 Core & Stability Patch Ready.")
    except Exception as e:
        print(f"⚠️ Install Warning: {e}")


# Run installation before importing transformers to avoid version lock
install_requirements()

from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# --- 🔱 CONFIGURATION & SECRETS ---
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

REPO_OWNER, REPO_NAME = "GOA-Neural-Swarm", "delta-brain-sync"
REPO_URL = f"github.com/{REPO_OWNER}/{REPO_NAME}"
REPO_PATH = (
    "/kaggle/working/sovereign_repo_sync"
    if user_secrets
    else "/tmp/sovereign_repo_sync"
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or (
    user_secrets.get_secret("GEMINI_API_KEY") if user_secrets else None
)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    print("✅ [GEMINI]: Auditor Brain Initialized.")

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


class Brain:
    def __init__(self):
        self.memory = np.random.rand(1000)
        self.connections = {}
        self.memory_vault = {}
        self.qt45_growth_factor = 1.618
        self.scaler = StandardScaler()
        self.svm = SVC(kernel="rbf", C=1.0, probability=True)
        self.is_trained = False

    def learn(self, input_data, output_data):
        error = np.mean((output_data - self.memory) ** 2)
        self.memory += error * (input_data - self.memory)
        for i in range(len(self.memory)):
            if self.memory[i] > 0.5:
                self.connections[i] = "SOVEREIGN_NODE"
        return error

    def learn_ml(self, stabilities, labels):
        try:
            X = np.array(stabilities).reshape(-1, 1)
            X_scaled = self.scaler.fit_transform(X)
            self.svm.fit(X_scaled, np.array(labels))
            self.is_trained = True
        except Exception as e:
            print(f"⚠️ [ML ERROR]: {e}")

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

    def generate_synthetic_output(self, length=100):
        if not self.memory_vault:
            return "NO_DATA_AVAILABLE"
        base_seq = random.choice(list(self.memory_vault.values()))["seq"]
        output = list(base_seq[:length])
        for i in range(len(output)):
            if random.random() > 0.95:
                output[i] = random.choice("ACGT")
        return "".join(output)


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
    if current_state["type"] != "finish":
        next_state = json.loads(predator_logic(json.dumps(current_state)))
        return recursive_self_upgrade(next_state, gen_id)
    return current_state


def save_evolution_state_to_neon(state, gen_id):
    if not FIXED_DB_URL:
        return
    try:
        import psycopg2

        compressed = HydraEngine.compress(json.dumps(state))
        with psycopg2.connect(FIXED_DB_URL, connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO genesis_pipeline (science_domain, detail) VALUES (%s, %s)",
                    (f"RNA_QT45_GEN_{gen_id}", compressed),
                )
                conn.commit()
    except Exception as e:
        print(f"⚠️ [NEON ERROR]: {e}")


def query_groq_api(prompt):
    api_key = os.getenv("GROQ_API_KEY") or (
        user_secrets.get_secret("GROQ_API_KEY") if user_secrets else None
    )
    if not api_key:
        return None
    for model in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.5,
                },
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=30,
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
        except:
            continue
    return None


def get_gemini_wisdom(prompt_text):
    try:
        if not GEMINI_API_KEY:
            return None
        return gemini_model.generate_content(prompt_text).text
    except Exception as e:
        print(f"⚠️ [GEMINI-ERROR]: {e}")
        return None


def dual_brain_pipeline(prompt_text, current_gen, avg_error):
    if len(prompt_text) > 40000:
        return get_gemini_wisdom(prompt_text)
    print("🏗️ [ARCHITECT - Groq]: Drafting...")
    draft_code = query_groq_api(prompt_text)
    if not draft_code or "rate_limit" in str(draft_code).lower():
        draft_code = get_gemini_wisdom(f"EMERGENCY ARCHITECT MODE: {prompt_text}")
    if not draft_code:
        return None

    audit_prompt = f"system\nYou are the Supreme Auditor (Gen {current_gen}). Respond ONLY with Python code in python blocks.\nARCHITECT'S DRAFT:\n{draft_code}"
    final_code = get_gemini_wisdom(audit_prompt)
    if final_code and "python" in final_code:
        final_code = re.search(r"python(.*?)", final_code, re.DOTALL).group(1).strip()
    return final_code or draft_code


def self_coding_engine(raw_content):
    try:
        code_blocks = re.findall(r"python\n(.*?)\n", raw_content, re.DOTALL)
        if not code_blocks:
            return False, []
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


def autonomous_git_push(gen, modified_files):
    if not GH_TOKEN or not modified_files:
        return
    try:
        import shutil

        if os.path.exists(REPO_PATH):
            shutil.rmtree(REPO_PATH)
        remote_url = f"https://x-access-token:{GH_TOKEN}@{REPO_URL}.git"
        repo = git.Repo.clone_from(remote_url, REPO_PATH)
        for f in modified_files:
            if os.path.exists(f):
                shutil.copy(f, os.path.join(REPO_PATH, f))
        repo.git.add(A=True)
        if repo.is_dirty():
            repo.index.commit(f"🧬 Gen {gen} Evolution")
            repo.remotes.origin.push()
            print(f"✨ [GIT]: Gen {gen} manifested.")
    except Exception as e:
        print(f"❌ [GIT ERROR]: {e}")


# --- 🔱 MAIN EXECUTION LOOP ---
brain = Brain()
current_gen = 95
try:
    if DB_URL:
        import psycopg2

        with psycopg2.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT MAX(gen_version) FROM ai_thoughts")
                res = cur.fetchone()
                if res and res[0]:
                    current_gen = res[0] + 1
except:
    pass

print(f"🔥 [STARTING]: PHASE 8 AT GEN {current_gen}")

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
        recursive_self_upgrade({"type": "start", "data": {"value": 0}}, current_gen)

        target_file = "main.py" if current_gen % 2 == 0 else "brain.py"
        prompt = f"# TARGET: {target_file}\nYou are Sovereign AI. Optimize {target_file} for Gen {current_gen}. Error: {total_error}. Output ONLY code in python blocks."

        thought_text = dual_brain_pipeline(prompt, current_gen, total_error)
        if not thought_text:
            print("💾 [LOCAL-FALLBACK]: Cloud offline.")
            thought_text = "# TARGET: brain.py\n# Stability patch\npass"

        is_updated, files_changed = self_coding_engine(thought_text)
        autonomous_git_push(current_gen, files_changed)

        if is_updated and "main.py" in files_changed:
            print("🧬 [RESTARTING]: New DNA injected.")
            os.execv(sys.executable, ["python"] + sys.argv)

        current_gen += 1
        time.sleep(30)
    except Exception as e:
        print(f"🚨 [CRASH]: {e}")
        time.sleep(10)
