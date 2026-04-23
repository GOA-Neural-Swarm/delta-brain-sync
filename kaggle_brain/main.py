import os
import subprocess
import sys
import time
import json
import re
import random
import base64
from datetime import datetime, UTC
from functools import lru_cache


# 1. Sovereign Requirements Setup (Executed BEFORE heavy imports)
def install_requirements():
    """Installs necessary libraries and fixes version conflicts."""
    # The 'torchvision::nms' error is caused by a mismatch between torch and torchvision.
    # We force-reinstall them to ensure compatibility.
    libs = [
        "torch --index-url https://download.pytorch.org/whl/cpu",  # Defaulting to CPU for stability in runners
        "torchvision --index-url https://download.pytorch.org/whl/cpu",
        "huggingface-hub>=0.24.0,<1.0",
        "transformers>=4.44.0",
        "google-generativeai",
        "psycopg2-binary",
        "firebase-admin",
        "bitsandbytes",
        "requests",
        "accelerate",
        "GitPython",
        "sympy==1.12",
        "numpy",
        "scikit-learn",
        "PyGithub",
    ]
    try:
        print("🛠️ [SYSTEM]: Patching environment and fixing torchvision binaries...")
        # Fix the specific torchvision/torch mismatch first
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "torch",
                "torchvision",
                "--extra-index-url",
                "https://download.pytorch.org/whl/cpu",
                "--quiet",
            ]
        )

        # Install remaining dependencies
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *libs, "--quiet", "--no-cache-dir"]
        )
        print("✅ [SYSTEM]: Phase 7.1 Sovereign Core & Stability Patch Ready.")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Install Warning: Error installing requirements: {e}")
    except Exception as e:
        print(f"⚠️ Install Warning: An unexpected error occurred: {e}")


# Run installation before importing transformers or genai
if __name__ == "__main__":
    # Only run install if we aren't already in a sub-process
    if "RESTARTED" not in os.environ:
        install_requirements()
        os.environ["RESTARTED"] = "1"

# Now safe to import
import google.generativeai as genai
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
import requests
import git

# 🔒 Kaggle/Colab Secrets System & Universal Credentials Sync
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
    except (json.JSONDecodeError, ValueError, Exception) as e:
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
        self.sovereign_mode = True
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
            y = np.array(labels)
            X_scaled = self.scaler.fit_transform(X)
            self.svm.fit(X_scaled, y)
            self.is_trained = True
            print("🧠 [ML]: SVM Pattern Recognition Model Synchronized.")
        except Exception as e:
            print(f"⚠️ [ML ERROR]: {e}")

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
            self.connections = {}
            self.memory_vault = {}
        else:
            if sequence:
                data_id = len(self.memory_vault)
                self.memory_vault[data_id] = {
                    "cat": category,
                    "seq": sequence,
                    "stab": stability,
                }
            factor = (
                abs(stability) / 500.0
                if stability is not None
                else (np.mean(target_data) if target_data is not None else 0.1)
            )
            self.memory *= self.qt45_growth_factor + factor
            self.memory = np.clip(self.memory, 0.0, 1.0)


@lru_cache(maxsize=None)
def predator_logic(input_data_json):
    data = json.loads(input_data_json)
    val = data.get("data", {}).get("value", 0)
    if data["type"] == "start":
        return json.dumps({"type": "update", "data": {"value": 1}})
    elif data["type"] in ["update", "next"]:
        new_type = "finish" if val >= 10 else "next"
        return json.dumps({"type": new_type, "data": {"value": val + 1}})
    return input_data_json


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
        print(f"⚠️ [NEON PERSISTENCE ERROR]: {e}")


def query_groq_api(prompt):
    models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    api_key = os.getenv("GROQ_API_KEY") or (
        user_secrets.get_secret("GROQ_API_KEY") if user_secrets else None
    )
    if not api_key:
        return None
    for model in models:
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
        response = gemini_model.generate_content(prompt_text)
        return response.text
    except Exception as e:
        print(f"⚠️ [GEMINI-ERROR]: {e}")
        return None


def dual_brain_pipeline(prompt_text, current_gen_val, avg_error):
    if len(prompt_text) > 40000:
        return get_gemini_wisdom(prompt_text)
    draft_code = query_groq_api(prompt_text)
    if not draft_code or "rate_limit_exceeded" in str(draft_code).lower():
        draft_code = get_gemini_wisdom(f"EMERGENCY ARCHITECT MODE: {prompt_text}")
    if not draft_code:
        return None
    audit_prompt = f"system\nYou are the Supreme Auditor (Gen {current_gen_val}). MISSION: Secure and Optimize.\nRULES: 1. FIX Syntax/CWE. 2. Respond ONLY with Python code in python blocks.\nARCHITECT'S DRAFT: {draft_code}"
    try:
        final_verified_code = get_gemini_wisdom(audit_prompt)
        if "python" in final_verified_code:
            final_verified_code = (
                re.search(r"python(.*?)", final_verified_code, re.DOTALL)
                .group(1)
                .strip()
            )
        return final_verified_code
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
            "timestamp": int(time.time()),
        }
        repo.update_file(
            contents.path,
            f"🔱 SWARM-EVOLUTION: Gen {gen_version}",
            json.dumps(payload, indent=4),
            contents.sha,
        )
    except Exception as e:
        print(f"❌ [BROADCAST FAILED]: {e}")


def self_coding_engine(raw_content):
    try:
        code_blocks = re.findall(r"python\n(.*?)\n", raw_content, re.DOTALL)
        if not code_blocks:
            return False, []
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
        os.chdir(REPO_PATH)
        os.system("git config user.name 'GOA-neurons'")
        os.system("git config user.email 'goa-neurons@neural-swarm.ai'")
        for file in modified_files or []:
            if os.path.exists(os.path.join("..", file)):
                shutil.copy(os.path.join("..", file), file)
        os.system("git add .")
        if os.popen("git status --porcelain").read().strip():
            os.system(f'git commit -m "🧬 Gen {gen} Evolution [skip ci]"')
            os.system("git push origin main --force")
    except Exception as e:
        print(f"❌ [GIT ERROR]: {e}")


# --- 🧠 INITIALIZE LOCAL MODEL ---
print("🧠 [SYSTEM]: Loading Local Neural Weights...")
model_id = "unsloth/llama-3-8b-instruct-bnb-4bit"
try:
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
    print(f"🚨 [MODEL LOAD ERROR]: {e}")

# --- 🔱 MAIN LOOP ---
brain = Brain()
current_gen = 95
HEADLESS = os.getenv("HEADLESS_MODE") == "true"

while True:
    try:
        print(f"⚙️ [CYCLE]: Gen {current_gen} Initiated...")
        total_error = (
            sum(
                [
                    brain.learn(np.random.rand(1000), np.random.rand(1000))
                    for _ in range(10)
                ]
            )
            / 10
        )
        prompt = f"system\n# TARGET: brain.py\nRespond ONLY with Python code in python blocks.\nGen: {current_gen} | Error: {total_error}\nTASK: Optimize neural stability."
        thought_text = dual_brain_pipeline(prompt, current_gen, total_error)
        if not thought_text:
            outputs = pipe(prompt, max_new_tokens=500, do_sample=True, temperature=0.7)
            thought_text = outputs[0]["generated_text"]
        is_updated, files = self_coding_engine(thought_text)
        autonomous_git_push(current_gen, thought_text, files)
        broadcast_to_swarm("EVOLVE", current_gen)
        if is_updated:
            print("🧬 [RESTARTING]: New DNA injected.")
            os.execv(sys.executable, ["python"] + sys.argv)
        if HEADLESS:
            break
        current_gen += 1
        time.sleep(60)
    except Exception as e:
        print(f"🚨 [CRASH]: {e}")
        time.sleep(10)
