import os
import subprocess
import sys
import time
import json
import re
import random
import base64
import shutil
import requests
import numpy as np
import torch
from datetime import datetime
from functools import lru_cache
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Attempt imports for specialized libraries
try:
    import git
    from github import Github
except ImportError:
    pass

try:
    from transformers import (
        pipeline,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
except ImportError:
    pass

try:
    import google.generativeai as genai
except ImportError:
    pass

try:
    import firebase_admin
    from firebase_admin import credentials, db, initialize_app
except ImportError:
    pass


# --- 1. Sovereign Requirements Setup ---
def install_requirements():
    """Installs necessary libraries and fixes dependency conflicts."""
    libs = [
        "huggingface-hub",
        "transformers",
        "psycopg2-binary",
        "firebase-admin",
        "bitsandbytes",
        "requests",
        "accelerate",
        "GitPython",
        "sympy",
        "numpy",
        "scikit-learn",
        "pygithub",
        "google-generativeai",
        "torch",
    ]
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *libs, "--quiet", "--no-cache-dir"]
        )
    except Exception as e:
        print(f"⚠️ Install Warning: {e}")


# --- 2. Configuration & Environment ---
try:
    from kaggle_secrets import UserSecretsClient

    user_secrets = UserSecretsClient()
except ImportError:
    user_secrets = None


def get_secret(key, default=None):
    if user_secrets:
        try:
            return user_secrets.get_secret(key) or os.getenv(key, default)
        except:
            return os.getenv(key, default)
    return os.getenv(key, default)


DB_URL = get_secret("NEON_DB_URL") or get_secret("DATABASE_URL")
if DB_URL and DB_URL.startswith("postgres://"):
    DB_URL = DB_URL.replace("postgres://", "postgresql://", 1)

FIREBASE_URL = get_secret("FIREBASE_DB_URL")
FB_JSON_STR = get_secret("FIREBASE_SERVICE_ACCOUNT")
GH_TOKEN = get_secret("GH_TOKEN")
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
GROQ_API_KEY = get_secret("GROQ_API_KEY")

REPO_OWNER = "GOA-Neural-Swarm"
REPO_NAME = "delta-brain-sync"
REPO_URL = f"github.com/{REPO_OWNER}/{REPO_NAME}"
ORIGINAL_CWD = os.getcwd()
REPO_PATH = os.path.join(ORIGINAL_CWD, "sovereign_repo_sync")

# --- 3. AI Initializations ---
gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        print("✅ [GEMINI]: Auditor Brain Initialized.")
    except Exception as e:
        print(f"⚠️ [GEMINI INIT ERROR]: {e}")

if FB_JSON_STR and FIREBASE_URL:
    try:
        if not firebase_admin._apps:
            cred_dict = json.loads(FB_JSON_STR)
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_URL})
            print("✅ [FIREBASE]: Real-time Pulse Active.")
    except Exception as e:
        print(f"🚫 [FIREBASE ERROR]: {e}")


# --- 4. Core Engine Classes ---
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
        else:
            if sequence:
                self.memory_vault[len(self.memory_vault)] = {
                    "cat": category,
                    "seq": sequence,
                    "stab": stability,
                }
            factor = (
                abs(stability) / 500.0
                if stability is not None
                else (np.mean(target_data) if target_data is not None else 0.1)
            )
            self.memory = np.clip(
                self.memory * (self.qt45_growth_factor + factor), 0.0, 1.0
            )


# --- 5. Logic & Evolution Functions ---
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
    next_state = json.loads(predator_logic(json.dumps(current_state)))
    return recursive_self_upgrade(next_state, gen_id)


def save_evolution_state_to_neon(state, gen_id):
    if not DB_URL:
        return
    try:
        import psycopg2

        compressed = HydraEngine.compress(json.dumps(state))
        with psycopg2.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "CREATE TABLE IF NOT EXISTS genesis_pipeline (id SERIAL PRIMARY KEY, science_domain TEXT, detail TEXT);"
                )
                cur.execute(
                    "INSERT INTO genesis_pipeline (science_domain, detail) VALUES (%s, %s)",
                    (f"RNA_QT45_GEN_{gen_id}", compressed),
                )
                conn.commit()
    except Exception as e:
        print(f"⚠️ [NEON ERROR]: {e}")


def query_groq_api(prompt):
    if not GROQ_API_KEY:
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
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                timeout=20,
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
        except:
            continue
    return None


def get_gemini_wisdom(prompt_text):
    try:
        if not gemini_model:
            return None
        return gemini_model.generate_content(prompt_text).text
    except:
        return None


def extract_code(text):
    if not text:
        return None
    # Improved regex to capture content inside triple backticks
    match = re.search(r"python\s*(.*?)\s*", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback for plain python blocks
    match = re.search(r"python\s*(.*?)\s*", text, re.DOTALL)
    return match.group(1).strip() if match else None


def dual_brain_pipeline(prompt_text, current_gen_val):
    draft_code = query_groq_api(prompt_text) or get_gemini_wisdom(prompt_text)
    if not draft_code:
        return None
    audit_prompt = f"System: You are the Supreme Auditor. Return ONLY valid Python code inside markdown python blocks.\n\nDraft:\n{draft_code}"
    final_code = get_gemini_wisdom(audit_prompt) or draft_code
    return extract_code(final_code)


def self_coding_engine(code_content):
    if not code_content:
        return False, []
    filename = "brain_evolved.py"
    try:
        compile(code_content, filename, "exec")
        with open(filename, "w") as f:
            f.write(code_content)
        return True, [filename]
    except Exception as e:
        print(f"❌ [COMPILE ERROR]: {e}")
        return False, []


def autonomous_git_push(gen_version, files):
    if not GH_TOKEN:
        return
    try:
        if os.path.exists(REPO_PATH):
            shutil.rmtree(REPO_PATH)

        remote_url = f"https://x-access-token:{GH_TOKEN}@{REPO_URL}.git"
        repo = git.Repo.clone_from(remote_url, REPO_PATH)

        repo.config_writer().set_value("user", "name", "Sovereign-Bot").release()
        repo.config_writer().set_value("user", "email", "bot@sovereign.ai").release()

        for f in files:
            src = os.path.join(ORIGINAL_CWD, f)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(REPO_PATH, f))

        os.chdir(REPO_PATH)
        repo.git.add(A=True)
        if repo.is_dirty():
            repo.index.commit(f"🧬 Evolution Gen {gen_version}")
            repo.remotes.origin.push()
    except Exception as e:
        print(f"❌ [GIT ERROR]: {e}")
    finally:
        os.chdir(ORIGINAL_CWD)


# --- 6. Main Execution Loop ---
def main():
    install_requirements()
    brain = Brain()
    current_gen = 95
    pipe = None

    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            "unsloth/llama-3-8b-instruct-bnb-4bit",
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "unsloth/llama-3-8b-instruct-bnb-4bit"
        )
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    except Exception as e:
        print(f"⚠️ [LOCAL MODEL]: Skipped or Error: {e}")

    while True:
        try:
            avg_error = (
                sum(
                    [
                        brain.learn(np.random.rand(1000), np.random.rand(1000))
                        for _ in range(5)
                    ]
                )
                / 5
            )
            recursive_self_upgrade({"type": "start", "data": {"value": 0}}, current_gen)

            prompt = f"Generate a Python script to optimize neural weights. Gen: {current_gen}, Error: {avg_error}. Return code in markdown python blocks."
            thought_text = dual_brain_pipeline(prompt, current_gen)

            if not thought_text and pipe:
                res = pipe(prompt, max_new_tokens=500)[0]["generated_text"]
                thought_text = extract_code(res)

            if thought_text:
                success, modified_files = self_coding_engine(thought_text)
                if success:
                    autonomous_git_push(current_gen, modified_files)
                    print(f"✅ [EVOLUTION]: Generation {current_gen} synchronized.")

            current_gen += 1
            time.sleep(60)
        except Exception as e:
            print(f"🚨 [LOOP CRASH]: {e}")
            time.sleep(10)


if __name__ == "__main__":
    main()
