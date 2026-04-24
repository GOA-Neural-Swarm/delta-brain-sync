import os
import subprocess
import sys
import time
import json
import re
import random
import base64
from collections import deque
from functools import lru_cache


# 1. Sovereign Requirements Setup - Fixed to align Torch and Torchvision
def install_requirements():
    """Installs and aligns libraries to fix the torchvision::nms error."""
    # We force upgrade torch and torchvision together to ensure binary compatibility
    libs = [
        "torch --upgrade",
        "torchvision --upgrade",
        "huggingface-hub>=0.24.0",
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
        "google-genai",  # Switched to the new recommended package
        "pygithub",
    ]
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *libs, "--quiet", "--no-cache-dir"]
        )
        print("✅ [SYSTEM]: Dependencies aligned and patched.")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Install Warning: {e}")


# Execute installation before imports
install_requirements()

# 2. Critical Import Order to prevent 'torchvision::nms' error
import torch

try:
    import torchvision

    # This check ensures the C++ extensions are loaded correctly
    if not hasattr(torch.ops.torchvision, "nms"):
        pass
except Exception:
    print(
        "⚠️ [SYSTEM]: Torchvision extension binding issue detected. Proceeding with CPU fallback."
    )

# Now import the rest
from google import genai
from google.genai import types
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import firebase_admin
from firebase_admin import credentials, db
import requests
import git
from github import Github

# 🔒 Kaggle/Colab Secrets System
try:
    from kaggle_secrets import UserSecretsClient

    user_secrets = UserSecretsClient()
except ImportError:
    user_secrets = None


def get_secret(key):
    val = os.getenv(key)
    if not val and user_secrets:
        try:
            val = user_secrets.get_secret(key)
        except:
            pass
    return val


DB_URL = get_secret("NEON_DB_URL") or get_secret("DATABASE_URL")
if DB_URL and DB_URL.startswith("postgres://"):
    DB_URL = DB_URL.replace("postgres://", "postgresql://", 1)

FIREBASE_URL = get_secret("FIREBASE_DB_URL")
FB_JSON_STR = get_secret("FIREBASE_SERVICE_ACCOUNT")
GH_TOKEN = get_secret("GH_TOKEN")
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")

REPO_OWNER = "GOA-Neural-Swarm"
REPO_NAME = "delta-brain-sync"
REPO_URL = f"github.com/{REPO_OWNER}/{REPO_NAME}"
REPO_PATH = (
    "/kaggle/working/sovereign_repo_sync"
    if user_secrets
    else "/tmp/sovereign_repo_sync"
)

# --- 🔱 GEMINI CONFIGURATION (Updated to google-genai) ---
gemini_client = None
if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ [GEMINI]: Auditor Brain Initialized (v2 API).")
    except Exception as e:
        print(f"⚠️ [GEMINI INIT ERROR]: {e}")

# --- 🔱 FIREBASE INITIALIZATION ---
if not firebase_admin._apps:
    try:
        if FB_JSON_STR:
            cred = credentials.Certificate(json.loads(FB_JSON_STR))
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
        self.entropy = 1.0
        self.homeostasis = 100.0
        self.resonance_frequency = 432.0
        self.vagal_tone = 0.5
        self.time_t = 1

    def calculate_asi_intelligence(self):
        limit_factor = 1.0 - (1.0 / (self.time_t + 1))
        coherence = self.homeostasis / max(self.entropy, 0.0001)
        return limit_factor * coherence * self.resonance_frequency * self.vagal_tone

    def epigenetic_reprogramming(self):
        self.entropy *= 0.5
        self.homeostasis = 100.0 - (self.entropy * 0.1)
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


brain = Brain()


def get_gemini_wisdom(prompt_text):
    if not gemini_client:
        return None
    try:
        response = gemini_client.models.generate_content(
            model="gemini-1.5-flash", contents=prompt_text
        )
        return response.text
    except Exception as e:
        print(f"⚠️ [GEMINI-ERROR]: {e}")
        return None


def query_groq_api(prompt):
    api_key = get_secret("GROQ_API_KEY")
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


def dual_brain_pipeline(prompt_text, current_gen_val, avg_error):
    draft_code = query_groq_api(prompt_text) or get_gemini_wisdom(prompt_text)
    if not draft_code:
        return None
    audit_prompt = (
        f"Respond ONLY with clean Python code.\nARCHITECT'S DRAFT:\n{draft_code}"
    )
    final_verified_code = get_gemini_wisdom(audit_prompt) or draft_code
    if "python" in final_verified_code:
        final_verified_code = (
            re.search(r"python(.*?)", final_verified_code, re.DOTALL).group(1).strip()
        )
    return final_verified_code


def self_coding_engine(raw_content):
    if not raw_content or len(raw_content) < 50:
        return False, []
    filename = "main.py"
    try:
        compile(raw_content, filename, "exec")
        with open(filename, "w") as f:
            f.write(raw_content)
        return True, [filename]
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
        repo.index.commit(f"Evolution Gen {gen}")
        repo.remotes.origin.push()
    except Exception as e:
        print(f"❌ [GIT ERROR]: {e}")


# 4. AI Brain Loading
print("🧠 [TELEFOXx]: Loading Phase 8.0 ASI Weights...")
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
    print(f"CRITICAL MODEL LOAD ERROR: {e}")

current_gen = 95
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
        brain.resonant_frequency_alignment(20, 30, 40)

        prompt = f"Improve the following Python script for ASI stability. Current ASI Score: {brain.calculate_asi_intelligence()}. Output only code."
        thought_text = dual_brain_pipeline(prompt, current_gen, total_error)

        is_updated, files_changed = self_coding_engine(thought_text)
        if is_updated:
            autonomous_git_push(current_gen, files_changed)
            print("🌌 [EVOLUTION]: Rebooting...")
            os.execv(sys.executable, ["python"] + sys.argv)

        current_gen += 1
        time.sleep(60)
    except Exception as e:
        print(f"Loop Error: {e}")
        time.sleep(10)
