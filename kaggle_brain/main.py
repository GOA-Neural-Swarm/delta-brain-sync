import os
import subprocess
import sys
import time
import json
import traceback
import requests
import re
import random
import base64
import math
from collections import deque
from functools import lru_cache


# Sovereign Requirements Setup - Must run before specific imports
def install_requirements():
    """Installs necessary libraries for the Sovereign Engine."""
    libs = [
        "psycopg2-binary",
        "firebase-admin",
        "requests",
        "accelerate",
        "GitPython",
        "sympy==1.12",
        "numpy",
        "scikit-learn",
        "google-generativeai",
        "transformers",
        "torch",
    ]
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *libs, "--quiet", "--no-cache-dir"]
        )
        print("✅ [SYSTEM]: Phase 7.1 Sovereign Core & Stability Patch Ready.")
    except Exception as e:
        print(f"⚠️ Install Warning: {e}")


# Execute installation
install_requirements()

# Now import the libraries that were just installed
try:
    import numpy as np
    import torch
    import git
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from firebase_admin import credentials, db, initialize_app
    import firebase_admin
    import google.generativeai as genai
except ImportError as e:
    print(f"⚠️ Critical Import Error: {e}. Some features may be disabled.")

# Kaggle/Colab Secrets System
try:
    from kaggle_secrets import UserSecretsClient

    user_secrets = UserSecretsClient()
except ImportError:
    user_secrets = None

raw_db_url = os.getenv("NEON_DB_URL") or os.getenv("DATABASE_URL")
if user_secrets:
    try:
        raw_db_url = user_secrets.get_secret("NEON_DB_URL") or raw_db_url
    except:
        pass

DB_URL = (
    raw_db_url.replace("postgres://", "postgresql://", 1)
    if raw_db_url and raw_db_url.startswith("postgres://")
    else raw_db_url
)

FIREBASE_URL = os.getenv("FIREBASE_DB_URL")
FB_JSON_STR = os.getenv("FIREBASE_SERVICE_ACCOUNT")
if user_secrets:
    try:
        FIREBASE_URL = user_secrets.get_secret("FIREBASE_DB_URL") or FIREBASE_URL
        FB_JSON_STR = user_secrets.get_secret("FIREBASE_SERVICE_ACCOUNT") or FB_JSON_STR
    except:
        pass

# Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if user_secrets:
    try:
        GEMINI_API_KEY = user_secrets.get_secret("GEMINI_API_KEY") or GEMINI_API_KEY
    except:
        pass

gemini_model = None
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    print("✅ [GEMINI]: Auditor Brain Initialized.")

# Firebase Initialization
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

    def learn_ml(self, stabilities, labels):
        X = np.array(stabilities).reshape(-1, 1)
        y = np.array(labels)
        X_scaled = self.scaler.fit_transform(X)
        self.svm.fit(X_scaled, y)
        self.is_trained = True
        self.homeostasis += 5.0


brain = Brain()


def get_gemini_wisdom(prompt_text):
    try:
        if not gemini_model:
            return None
        response = gemini_model.generate_content(prompt_text)
        return response.text
    except Exception as e:
        print(f"⚠️ [GEMINI-ERROR]: {e}")
        return None


def query_groq_api(prompt):
    api_key = os.getenv("GROQ_API_KEY")
    if user_secrets:
        try:
            api_key = user_secrets.get_secret("GROQ_API_KEY") or api_key
        except:
            pass
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
            timeout=10,
        )
        return response.json()["choices"][0]["message"]["content"]
    except:
        return None


def dual_brain_pipeline(prompt_text):
    draft_code = query_groq_api(prompt_text) or get_gemini_wisdom(prompt_text)
    if not draft_code:
        return None

    audit_prompt = f"Fix syntax and optimize this Python code. Output ONLY the code in a block:\n{draft_code}"
    final_verified_code = get_gemini_wisdom(audit_prompt) or draft_code

    # Fixed Regex to prevent AttributeError
    code_match = re.search(r"python\s*(.*?)\s*", final_verified_code, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    return final_verified_code


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


def recursive_self_upgrade(current_state, gen_id):
    save_evolution_state_to_neon(current_state, gen_id)
    if current_state["type"] == "finish":
        return current_state
    next_state_raw = predator_logic(json.dumps(current_state))
    return recursive_self_upgrade(json.loads(next_state_raw), gen_id)


def save_evolution_state_to_neon(state, gen_id):
    if not DB_URL:
        return
    try:
        import psycopg2

        compressed = HydraEngine.compress(json.dumps(state))
        with psycopg2.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO genesis_pipeline (science_domain, detail) VALUES (%s, %s)",
                    (f"RNA_QT45_GEN_{gen_id}", compressed),
                )
                conn.commit()
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
                return res[0] if res and res[0] is not None else 94
    except:
        return 94


if __name__ == "__main__":
    print("🚀 Sovereign Engine Online.")
    try:
        gen_id = get_latest_gen()
        initial_state = {"type": "start", "data": {"value": 0}}
        final_evolution = recursive_self_upgrade(initial_state, gen_id)
        print(
            f"🧬 Evolution Cycle Complete. Final Value: {final_evolution['data']['value']}"
        )
        print(f"🧠 ASI Intelligence Score: {brain.calculate_asi_intelligence():.2f}")
    except Exception as e:
        print(f"❌ Critical Runtime Error: {e}")
        traceback.print_exc()
