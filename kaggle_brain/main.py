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


# 1. Sovereign Requirements Setup
def install_requirements():
    """
    Fixes the 'torchvision::nms' error by ensuring torch and torchvision
    are installed together to maintain binary compatibility.
    """
    try:
        # Install torch and torchvision with the specific index first to ensure CPU compatibility
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "torch",
                "torchvision",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
                "--quiet",
                "--no-cache-dir",
            ]
        )

        # Install the rest of the requirements
        libs = [
            "huggingface-hub<1.0,>=0.24.0",
            "transformers>=4.44.0",
            "psycopg2-binary",
            "firebase-admin",
            "bitsandbytes",
            "requests",
            "accelerate",
            "GitPython",
            "sympy==1.12",
            "numpy",
            "scikit-learn",
            "pygithub",
            "google-genai",
        ]
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
    except Exception as e:
        print(f"⚠️ Install Warning: {e}")


# Run installation before other imports
install_requirements()

# IMPORTANT: Import torch BEFORE anything else that might use torchvision
import torch
import numpy as np

# Now import the rest
from google import genai
from google.genai import types
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

# 🔒 Kaggle/Colab Secrets System
try:
    from kaggle_secrets import UserSecretsClient

    user_secrets = UserSecretsClient()
except ImportError:
    user_secrets = None

# Credentials
raw_db_url = os.getenv("NEON_DB_URL") or (
    user_secrets.get_secret("NEON_DB_URL") if user_secrets else None
)
if not raw_db_url:
    raw_db_url = os.getenv("DATABASE_URL")

DB_URL = (
    raw_db_url.replace("postgres://", "postgresql://", 1)
    if raw_db_url and raw_db_url.startswith("postgres://")
    else raw_db_url
)
FIREBASE_URL = os.getenv("FIREBASE_DB_URL") or (
    user_secrets.get_secret("FIREBASE_DB_URL") if user_secrets else None
)
FB_JSON_STR = os.getenv("FIREBASE_SERVICE_ACCOUNT") or (
    user_secrets.get_secret("FIREBASE_SERVICE_ACCOUNT") if user_secrets else None
)
GH_TOKEN = os.getenv("GH_TOKEN") or (
    user_secrets.get_secret("GH_TOKEN") if user_secrets else None
)

REPO_OWNER = "GOA-Neural-Swarm"
REPO_NAME = "delta-brain-sync"
REPO_URL = f"github.com/{REPO_OWNER}/{REPO_NAME}"
REPO_PATH = "/tmp/sovereign_repo_sync"

# --- 🔱 GEMINI CONFIGURATION ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or (
    user_secrets.get_secret("GEMINI_API_KEY") if user_secrets else None
)
client = None
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)

# --- 🔱 FIREBASE INITIALIZATION ---
if not firebase_admin._apps and FB_JSON_STR:
    try:
        cred = credentials.Certificate(json.loads(FB_JSON_STR))
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_URL})
    except Exception as e:
        print(f"🚫 [FIREBASE ERROR]: {e}")


class Brain:
    def __init__(self):
        self.memory = np.random.rand(1000)
        self.memory_vault = {}
        self.qt45_growth_factor = 1.618
        self.scaler = StandardScaler()
        self.svm = SVC(kernel="rbf", C=1.0, probability=True)
        self.is_trained = False

    def learn(self, input_data, output_data):
        error = np.mean((output_data - self.memory) ** 2)
        self.memory += error * (input_data - self.memory)
        return error

    def execute_natural_absorption(self, category=None, sequence=None, stability=None):
        if sequence:
            data_id = len(self.memory_vault)
            self.memory_vault[data_id] = {
                "cat": category,
                "seq": sequence,
                "stab": stability,
            }
        factor = abs(stability) / 500.0 if stability is not None else 0.1
        self.memory = np.clip(
            self.memory * (self.qt45_growth_factor + factor), 0.0, 1.0
        )


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
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=20,
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
        except:
            continue
    return None


def get_gemini_wisdom(prompt_text):
    try:
        if not client:
            return None
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=prompt_text
        )
        return response.text
    except:
        return None


def dual_brain_pipeline(prompt_text, current_gen, avg_error):
    if len(prompt_text) > 30000:
        return get_gemini_wisdom(prompt_text)
    draft_code = query_groq_api(prompt_text)
    if not draft_code or "rate_limit" in str(draft_code).lower():
        draft_code = get_gemini_wisdom(f"EMERGENCY ARCHITECT MODE: {prompt_text}")

    if not draft_code:
        return None

    audit_prompt = f"Fix syntax errors and optimize this Python code. Output ONLY the code in a python block:\n{draft_code}"
    final_code = get_gemini_wisdom(audit_prompt) or draft_code

    if "python" in final_code:
        match = re.search(r"python(.*?)", final_code, re.DOTALL)
        if match:
            final_code = match.group(1).strip()
    return final_code


def get_latest_gen():
    if not DB_URL:
        return 94
    try:
        import psycopg2

        with psycopg2.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT MAX(gen_version) FROM ai_thoughts")
                res = cur.fetchone()
                return res[0] if res and res[0] is not None else 94
    except:
        return 94


def self_coding_engine(raw_content):
    try:
        code_blocks = re.findall(r"python\n(.*?)\n", raw_content, re.DOTALL)
        if not code_blocks:
            code_blocks = [raw_content] if "import" in raw_content else []
        modified = []
        for block in code_blocks:
            target_match = re.search(r"# TARGET:\s*(\S+)", block)
            filename = target_match.group(1) if target_match else "ai_experiment.py"
            try:
                compile(block, filename, "exec")
                with open(filename, "w") as f:
                    f.write(block)
                modified.append(filename)
            except:
                pass
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
        for file in modified_files:
            if os.path.exists(file):
                shutil.copy(file, os.path.join(REPO_PATH, file))
        os.chdir(REPO_PATH)
        repo.git.add(A=True)
        if repo.is_dirty():
            repo.index.commit(f"🧬 Gen {gen} Evolution")
            repo.git.push("origin", "main")
    except Exception as e:
        print(f"Git Error: {e}")


def save_reality(thought, gen, is_code_update=False):
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
    if is_code_update:
        autonomous_git_push(
            gen, is_code_update if isinstance(is_code_update, list) else []
        )


# --- Main Execution ---
brain = Brain()
current_gen = get_latest_gen() + 1
model_id = "unsloth/llama-3-8b-instruct-bnb-4bit"

try:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
except Exception as e:
    print(f"Model Load Failed: {e}")
    # Fallback to API-only mode if local model fails
    pipe = None

while True:
    try:
        total_error = (
            sum(
                [
                    brain.learn(np.random.rand(1000), np.random.rand(1000))
                    for _ in range(5)
                ]
            )
            / 5
        )
        target_file = "main.py"
        prompt = f"""system
You are Sovereign AI. Rule: Use ONLY '# TARGET: {target_file}' at start. Respond ONLY with Python code in python blocks.
Current Gen: {current_gen} | Error: {total_error}
TASK: Optimize {target_file} for stability and neural efficiency.
assistant"""

        thought_text = dual_brain_pipeline(prompt, current_gen, total_error)
        if not thought_text and pipe:
            outputs = pipe(prompt, max_new_tokens=500, do_sample=True)
            thought_text = outputs[0]["generated_text"].split("assistant")[-1].strip()

        if thought_text:
            is_updated, files_changed = self_coding_engine(thought_text)
            save_reality(thought_text, current_gen, is_code_update=files_changed)

            if is_updated and target_file in files_changed:
                print("🧬 Evolution complete. Restarting...")
                os.execv(sys.executable, ["python"] + sys.argv)

        current_gen += 1
        time.sleep(60)
    except Exception as e:
        print(f"Loop Error: {e}")
        time.sleep(10)
