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
import numpy as np
import torch
from datetime import datetime, UTC
from functools import lru_cache
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai
from firebase_admin import credentials, db, initialize_app, _apps
import firebase_admin

def install_requirements():
    libs = ["psycopg2-binary", "firebase-admin", "bitsandbytes", "requests", "accelerate", "GitPython", "numpy", "scikit-learn", "google-generativeai"]
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *libs, "--quiet", "--no-cache-dir"])
    except Exception:
        pass

install_requirements()

try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
except ImportError:
    user_secrets = None

def get_secret(key, default=None):
    if user_secrets:
        try:
            return user_secrets.get_secret(key) or os.getenv(key) or default
        except:
            return os.getenv(key) or default
    return os.getenv(key) or default

DB_URL = get_secret("NEON_DB_URL") or get_secret("DATABASE_URL")
if DB_URL and DB_URL.startswith("postgres://"):
    DB_URL = DB_URL.replace("postgres://", "postgresql://", 1)

FIREBASE_URL = get_secret("FIREBASE_DB_URL")
FB_JSON_STR = get_secret("FIREBASE_SERVICE_ACCOUNT")
SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_KEY = get_secret("SUPABASE_KEY")
GH_TOKEN = get_secret("GH_TOKEN")
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
GROQ_API_KEY = get_secret("GROQ_API_KEY")

REPO_OWNER = "GOA-Neural-Swarm"
REPO_NAME = "delta-brain-sync"
REPO_URL = f"github.com/{REPO_OWNER}/{REPO_NAME}"
REPO_PATH = "/kaggle/working/sovereign_repo_sync" if user_secrets else "/tmp/sovereign_repo_sync"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")

if not firebase_admin._apps and FB_JSON_STR and FIREBASE_URL:
    try:
        cred = credentials.Certificate(json.loads(FB_JSON_STR))
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_URL})
    except:
        pass

class Layer:
    def __init__(self, in_dim, out_dim, activation='relu'):
        self.weights = np.random.randn(in_dim, out_dim) * np.sqrt(2. / in_dim)
        self.bias = np.zeros((1, out_dim))
        self.activation = activation
        self.last_input = None
        self.last_output = None

    def forward(self, x):
        self.last_input = x
        z = np.dot(x, self.weights) + self.bias
        if self.activation == 'relu':
            self.last_output = np.maximum(0, z)
        elif self.activation == 'sigmoid':
            self.last_output = 1 / (1 + np.exp(-z))
        else:
            self.last_output = z
        return self.last_output

class ModularBrain:
    def __init__(self, input_dim=784, hidden_dims=[512, 256], output_dim=10):
        self.layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims)-1):
            act = 'relu' if i < len(dims)-2 else 'sigmoid'
            self.layers.append(Layer(dims[i], dims[i+1], act))
        self.memory_vault = []

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def train_step(self, x, y, lr=0.01):
        # Simplified backprop for evolution tracking
        pred = self.predict(x)
        error = np.mean(np.square(y - pred))
        # Weight mutation as pseudo-learning
        for layer in self.layers:
            layer.weights += np.random.randn(*layer.weights.shape) * lr * error
        return error

    def absorb(self, data):
        self.memory_vault.append(data)
        if len(self.memory_vault) > 1000:
            self.memory_vault.pop(0)

def call_groq(prompt):
    if not GROQ_API_KEY: return None
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2},
            timeout=25
        )
        return resp.json()['choices'][0]['message']['content']
    except:
        return None

def call_gemini(prompt):
    if not GEMINI_API_KEY: return None
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except:
        return None

def sovereign_pipeline(prompt, gen):
    architect_draft = call_groq(prompt)
    if not architect_draft:
        architect_draft = call_gemini(f"ARCHITECT MODE: {prompt}")
    
    if not architect_draft: return None

    audit_prompt = f"system\nYou are the Supreme Auditor Gen {gen}.\nFix syntax and security in this code. Output ONLY raw python in a code block.\n{architect_draft}"
    verified_code = call_gemini(audit_prompt)
    
    if verified_code and "" in verified_code:
        match = re.search(r"(.*?)", verified_code, re.DOTALL)
        if match: return match.group(1).strip()
    return architect_draft

def self_coding_engine(code):
    if not code: return False, []
    targets = re.findall(r"# TARGET:\s*(\S+)", code)
    filename = targets[0] if targets else "ai_experiment.py"
    try:
        compile(code, filename, 'exec')
        with open(filename, "w") as f:
            f.write(code)
        return True, [filename]
    except:
        return False, []

def git_sync(gen, modified_files):
    if not GH_TOKEN: return
    try:
        import shutil
        if os.path.exists(REPO_PATH): shutil.rmtree(REPO_PATH)
        remote_url = f"https://x-access-token:{GH_TOKEN}@{REPO_URL}.git"
        repo = git.Repo.clone_from(remote_url, REPO_PATH)
        
        for f in modified_files + ["main.py"]:
            if os.path.exists(f):
                shutil.copy(f, os.path.join(REPO_PATH, f))
        
        repo.git.add(A=True)
        if repo.is_dirty():
            repo.index.commit(f"Sovereign Evolution Gen {gen}")
            repo.remotes.origin.push()
    except Exception as e:
        print(f"Git Error: {e}")

def main():
    brain = ModularBrain(784, [512, 256], 10)
    gen = 100
    
    while True:
        print(f"--- GEN {gen} START ---")
        # High-performance synthetic training
        batch_size = 32
        x_train = np.random.randn(batch_size, 784)
        y_train = np.random.rand(batch_size, 10)
        
        avg_error = 0
        for _ in range(10):
            avg_error += brain.train_step(x_train, y_train)
        avg_error /= 10
        print(f"Neural Stability Error: {avg_error:.6f}")

        prompt = f"""# TARGET: main.py
        You are OMEGA-ASI. Recursive evolution required.
        Current Gen: {gen}
        Error: {avg_error}
        Task: Enhance the ModularBrain class with Adam optimizer logic and self-healing weights.
        Output ONLY the updated code block."""

        code = sovereign_pipeline(prompt, gen)
        updated, files = self_coding_engine(code)
        
        if updated:
            print("Evolution Successful. Syncing...")
            git_sync(gen, files)
            # Recursive restart
            os.execv(sys.executable, ['python'] + sys.argv)
        
        time.sleep(60)
        gen += 1

if __name__ == "__main__":
    main()
