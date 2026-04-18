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
from datetime import datetime
from functools import lru_cache


def install_requirements():
    libs = [
        "torch",
        "torchvision",
        "huggingface-hub<1.0",
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
        "google-genai",
        "pygithub",
    ]
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "--quiet"]
        )
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                *libs,
                "--quiet",
                "--no-cache-dir",
            ]
        )
        print("Sovereign Core & Stability Patch Ready.")
    except subprocess.CalledProcessError as e:
        print(f"Install Warning: Error installing requirements: {e}")
    except Exception as e:
        print(f"Install Warning: An unexpected error occurred: {e}")


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
            print("SVM Pattern Recognition Model Synchronized.")
        except Exception as e:
            print(f"ML ERROR: {e}")

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
            factor = abs(stability) / 500.0 if stability is not None else 0.1
            self.memory *= self.qt45_growth_factor + factor
            self.memory = np.clip(self.memory, 0.0, 1.0)

    def generate_synthetic_output(self, length=100):
        if not self.memory_vault:
            return "NO_DATA_AVAILABLE"
        base_data = random.choice(list(self.memory_vault.values()))
        base_seq = base_data["seq"]
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
    if not os.getenv("DATABASE_URL"):
        return
    try:
        import psycopg2

        compressed = HydraEngine.compress(json.dumps(state))
        with psycopg2.connect(os.getenv("DATABASE_URL"), connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO genesis_pipeline (science_domain, detail) VALUES (%s, %s)",
                    (f"RNA_QT45_GEN_{gen_id}", compressed),
                )
                conn.commit()
    except Exception as e:
        print(f"NEON PERSISTENCE ERROR: {e}")


def query_groq_api(prompt):
    api_key = os.getenv("GROQ_API_KEY")
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
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        genai.initialize(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        response = gemini_model.generate_content(prompt_text)
        return response.text
    except Exception as e:
        return None


def dual_brain_pipeline(prompt_text, current_gen_val, avg_error):
    draft_code = query_groq_api(prompt_text)
    if draft_code:
        return draft_code
    return get_gemini_wisdom(f"EMERGENCY ARCHITECT MODE: {prompt_text}")


def broadcast_to_swarm(command, gen_version):
    if not os.getenv("GH_TOKEN"):
        return
    try:
        g = Github(os.getenv("GH_TOKEN"))
        repo = g.get_repo("GOA-Neural-Swarm/sub-node-logic")
        contents = repo.get_contents("instruction.json")
        payload = {
            "command": command,
            "gen_version": gen_version,
            "timestamp": int(time.time()),
        }
        repo.update_file(
            contents.path,
            f"SWARM-EVOLUTION: Gen {gen_version}",
            json.dumps(payload, indent=4),
            contents.sha,
        )
    except Exception as e:
        print(f"BROADCAST FAILED: {e}")


def get_latest_gen():
    if not os.getenv("DATABASE_URL"):
        return 94
    try:
        import psycopg2

        with psycopg2.connect(os.getenv("DATABASE_URL")) as conn:
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
    if not os.getenv("GH_TOKEN"):
        return
    try:
        import shutil

        if os.path.exists("/kaggle/working/sovereign_repo_sync"):
            shutil.rmtree("/kaggle/working/sovereign_repo_sync")
        remote_url = f"https://x-access-token:{os.getenv('GH_TOKEN')}@github.com/GOA-Neural-Swarm/delta-brain-sync.git"
        repo = git.Repo.clone_from(remote_url, "/kaggle/working/sovereign_repo_sync")
        os.chdir("/kaggle/working/sovereign_repo_sync")
        os.system("git config user.name 'GOA-neurons'")
        os.system("git config user.email 'goa-neurons@neural-swarm.ai'")
        for file in modified_files or []:
            if os.path.exists(os.path.join("..", file)):
                shutil.copy(os.path.join("..", file), file)
        os.system("git add .")
        if os.popen("git status --porcelain").read().strip():
            os.system(f'git commit -m "Gen {gen} Evolution"')
            os.system("git push origin main --force")
    except Exception as e:
        print(f"GIT ERROR: {e}")


install_requirements()

import google.genai as genai
import numpy as np
import torch
import torchvision
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

brain = Brain()
current_gen = get_latest_gen() + 1
HEADLESS = os.getenv("HEADLESS_MODE") == "true"

print("Loading Local Neural Engine...")
model_id = "unsloth/llama-3-8b-instruct-bnb-4bit"
try:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
except Exception as e:
    print(f"Local Model Load Failed: {e}")
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
        prompt = f"# TARGET: brain.py\nOptimize neural processing for Gen {current_gen}. Error: {total_error}"
        thought_text = dual_brain_pipeline(prompt, current_gen, total_error)
        if not thought_text and pipe:
            outputs = pipe(prompt, max_new_tokens=500, do_sample=True)
            thought_text = outputs[0]["generated_text"]
        is_updated, files_changed = self_coding_engine(thought_text)
        autonomous_git_push(current_gen, thought_text, files_changed)
        broadcast_to_swarm("EVOLVE", current_gen)
        if is_updated:
            print("RESTARTING: New DNA injected.")
            os.execv(sys.executable, ["python"] + sys.argv)
        if HEADLESS:
            break
        current_gen += 1
        time.sleep(60)
    except Exception as e:
        print(f"LOOP ERROR: {e}")
        time.sleep(10)
