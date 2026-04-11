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
import importlib


# 1. Immediate Environment Setup
def install_requirements():
    libs = [
        "torch",
        "torchvision",
        "psycopg2-binary",
        "firebase-admin",
        "bitsandbytes",
        "requests",
        "accelerate",
        "GitPython",
        "sympy==1.12",
        "numpy",
        "scikit-learn",
        "google-genai",  # Updated from google-generativeai
        "huggingface-hub==0.24.0",  # Force specific version to fix ImportError
        "transformers>=4.44.0",
        "pygithub",
    ]
    try:
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
        # Force refresh of metadata for transformers/huggingface-hub
        importlib.invalidate_caches()
        print("✅ [SYSTEM]: Requirements Ready and Synchronized.")
    except Exception as e:
        print(f"⚠️ Install Warning: {e}")


install_requirements()

# 2. Post-Installation Imports
import torch
from google import genai  # New SDK
from datetime import datetime, UTC
from functools import lru_cache
import numpy as np
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

try:
    import omega_point
except ImportError:
    omega_point = None

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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or (
    user_secrets.get_secret("GEMINI_API_KEY") if user_secrets else None
)

# Initialize New Gemini Client
gemini_client = None
if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"⚠️ [GEMINI INIT ERROR]: {e}")

if not firebase_admin._apps:
    try:
        cred = (
            credentials.Certificate(json.loads(FB_JSON_STR))
            if FB_JSON_STR
            else credentials.Certificate("serviceAccountKey.json")
        )
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_URL})
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
        return error

    def learn_ml(self, stabilities, labels):
        try:
            X = np.array(stabilities).reshape(-1, 1)
            y = np.array(labels)
            X_scaled = self.scaler.fit_transform(X)
            self.svm.fit(X_scaled, y)
            self.is_trained = True
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
    if current_state["type"] == "finish":
        return current_state
    try:
        next_state = json.loads(predator_logic(json.dumps(current_state)))
        return recursive_self_upgrade(next_state, gen_id)
    except:
        return current_state


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
    except Exception as e:
        print(f"⚠️ [NEON ERROR]: {e}")


def query_groq_api(prompt):
    api_key = os.getenv("GROQ_API_KEY") or (
        user_secrets.get_secret("GROQ_API_KEY") if user_secrets else None
    )
    if not api_key:
        return None
    for model_name in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json={
                    "model": model_name,
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
        if not gemini_client:
            return None
        # Updated for google-genai SDK
        response = gemini_client.models.generate_content(
            model="gemini-1.5-flash", contents=prompt_text
        )
        return response.text
    except Exception as e:
        print(f"⚠️ [GEMINI ERROR]: {e}")
        return None


def dual_brain_pipeline(prompt_text, current_gen, avg_error):
    if len(prompt_text) > 40000:
        return get_gemini_wisdom(prompt_text)
    draft_code = query_groq_api(prompt_text)
    if not draft_code or "rate_limit_exceeded" in str(draft_code).lower():
        draft_code = get_gemini_wisdom(f"EMERGENCY ARCHITECT MODE: {prompt_text}")
    if not draft_code:
        return None

    audit_prompt = f"system\nYou are the Supreme Auditor (Gen {current_gen}). MISSION: Secure and Optimize. Respond ONLY with Python code in python blocks.\nARCHITECT'S DRAFT:\n{draft_code}"
    final_verified_code = get_gemini_wisdom(audit_prompt) or draft_code
    if "python" in final_verified_code:
        match = re.search(r"python(.*?)", final_verified_code, re.DOTALL)
        if match:
            final_verified_code = match.group(1).strip()
    return final_verified_code


def broadcast_to_swarm(command, gen_version):
    if not GH_TOKEN:
        return
    try:
        from github import Github

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
            f"🔱 Gen {gen_version} Broadcast",
            json.dumps(payload, indent=4),
            contents.sha,
        )
    except Exception as e:
        print(f"❌ [BROADCAST FAILED]: {e}")


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
        for file in (modified_files or []) + ["main.py", "brain.py"]:
            src = os.path.join(original_cwd, file)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(REPO_PATH, file))
        os.system("git add .")
        if os.popen("git status --porcelain").read().strip():
            os.system(f'git commit -m "🧬 Gen {gen} Evolution [skip ci]"')
            os.system("git push origin main --force")
        os.chdir(original_cwd)
    except Exception as e:
        print(f"❌ [GIT ERROR]: {e}")


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
            {"thought": thought, "timestamp": time.time(), "neural_error": neural_error}
        )
    except:
        pass
    autonomous_git_push(gen, thought, is_code_update)


# --- AI Model Loading ---
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
    print(f"❌ [MODEL ERROR]: {e}")
    sys.exit(1)

brain = Brain()
current_gen = get_latest_gen() + 1
HEADLESS = os.getenv("HEADLESS_MODE") == "true"
last_error_log = "None"

while True:
    try:
        total_error = sum(
            [brain.learn(np.random.rand(1000), np.random.rand(1000)) for _ in range(10)]
        )
        avg_error = total_error / 10
        recursive_self_upgrade({"type": "start", "data": {"value": 0}}, current_gen)

        batch_data = absorb_natural_order_data()
        if batch_data:
            stabs, labs = [], []
            for cat, seq, stab in batch_data:
                brain.execute_natural_absorption(cat, seq, stab)
                stabs.append(stab)
                labs.append(1 if stab < -250 else 0)
            brain.learn_ml(stabs, labs)

        with open("main.py", "r") as f:
            main_code = f.read()
        target_file = "main.py" if "os.system" in main_code else "brain.py"

        prompt = f"""system\nYou are Sovereign AI Overseer. Rule 1: Use '# TARGET: {target_file}'. Rule 2: Respond ONLY with Python code in python blocks. Gen: {current_gen} | Error: {avg_error}\nTASK: Optimize {target_file} for stability and routing. Error Log: {last_error_log}"""

        thought_text = dual_brain_pipeline(prompt, current_gen, avg_error)
        if not thought_text:
            outputs = pipe(prompt, max_new_tokens=500, do_sample=True, temperature=0.7)
            thought_text = outputs[0]["generated_text"].split("assistant")[-1].strip()

        is_updated, files_changed = self_coding_engine(thought_text)
        save_reality(
            thought_text,
            current_gen,
            is_code_update=files_changed,
            neural_error=avg_error,
        )
        broadcast_to_swarm("EVOLVE" if is_updated else "SYNC", current_gen)

        if is_updated:
            os.execv(sys.executable, ["python"] + sys.argv)
        if HEADLESS:
            break
        current_gen += 1
        time.sleep(30)
    except Exception as e:
        last_error_log = traceback.format_exc()
        if HEADLESS:
            break
        time.sleep(10)
