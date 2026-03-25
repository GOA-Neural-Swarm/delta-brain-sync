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
from datetime import datetime, UTC
from functools import lru_cache

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

# --- CONFIGURATION ---
DB_URL = (get_secret("NEON_DB_URL") or get_secret("DATABASE_URL") or "").replace("postgres://", "postgresql://", 1)
FIREBASE_URL = get_secret("FIREBASE_DB_URL")
FB_JSON_STR = get_secret("FIREBASE_SERVICE_ACCOUNT")
SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_KEY = get_secret("SUPABASE_KEY")
GH_TOKEN = get_secret("GH_TOKEN")
GROQ_API_KEY = get_secret("GROQ_API_KEY")
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")

REPO_OWNER = "GOA-Neural-Swarm"
REPO_NAME = "delta-brain-sync"
REPO_URL = f"github.com/{REPO_OWNER}/{REPO_NAME}"
REPO_PATH = "/kaggle/working/sovereign_repo_sync" if user_secrets else "/tmp/sovereign_repo_sync"

# --- MODULAR NEURAL ARCHITECTURE (784 FEATURES) ---
class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    def forward(self, input_data): raise NotImplementedError
    def backward(self, output_error, learning_rate): raise NotImplementedError

class DenseLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.bias = np.zeros((1, output_size))
    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
    def backward(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * np.sum(output_error, axis=0, keepdims=True)
        return input_error

class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output
    def backward(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error

def relu(x): return np.maximum(0, x)
def relu_prime(x): return (x > 0).astype(float)
def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
def sigmoid_prime(x): 
    s = sigmoid(x)
    return s * (1 - s)

class SovereignNetwork:
    def __init__(self):
        self.layers = [
            DenseLayer(784, 256),
            ActivationLayer(relu, relu_prime),
            DenseLayer(256, 64),
            ActivationLayer(relu, relu_prime),
            DenseLayer(64, 10),
            ActivationLayer(sigmoid, sigmoid_prime)
        ]
    def predict(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output
    def train(self, x_train, y_train, epochs, lr):
        for epoch in range(epochs):
            err = 0
            for x, y in zip(x_train, y_train):
                output = self.predict(x)
                err += np.mean(np.power(y - output, 2))
                error = 2 * (output - y) / y.size
                for layer in reversed(self.layers):
                    error = layer.backward(error, lr)
            err /= len(x_train)
            if epoch % 10 == 0: yield err

# --- REDUNDANT LLM LOGIC ---
class LLMRegistry:
    @staticmethod
    def query_groq(prompt):
        if not GROQ_API_KEY: return None
        try:
            resp = requests.post("https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2},
                timeout=20)
            return resp.json()['choices'][0]['message']['content'] if resp.status_code == 200 else None
        except: return None

    @staticmethod
    def query_gemini(prompt):
        if not GEMINI_API_KEY: return None
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={GEMINI_API_KEY}"
            resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=20)
            return resp.json()['candidates'][0]['content']['parts'][0]['text'] if resp.status_code == 200 else None
        except: return None

    @classmethod
    def evolve_logic(cls, prompt):
        res = cls.query_groq(prompt)
        if not res: res = cls.query_gemini(prompt)
        return res

# --- PERSISTENCE & SYNC ---
def broadcast_swarm(cmd, gen):
    if not GH_TOKEN: return
    try:
        from github import Github
        g = Github(GH_TOKEN)
        repo = g.get_repo("GOA-Neural-Swarm/sub-node-logic")
        f = repo.get_contents("instruction.json")
        payload = {"command": cmd, "gen": gen, "ts": int(time.time())}
        repo.update_file(f.path, f"Gen {gen} Sync", json.dumps(payload), f.sha)
    except: pass

def git_sync(gen, files):
    if not GH_TOKEN: return
    try:
        if os.path.exists(REPO_PATH): subprocess.run(["rm", "-rf", REPO_PATH])
        remote = f"https://x-access-token:{GH_TOKEN}@{REPO_URL}.git"
        repo = git.Repo.clone_from(remote, REPO_PATH)
        for f in files:
            if os.path.exists(f): 
                import shutil
                shutil.copy(f, os.path.join(REPO_PATH, f))
        os.chdir(REPO_PATH)
        repo.git.add(A=True)
        if repo.is_dirty():
            repo.index.commit(f"Evolution Gen {gen}")
            repo.git.push("origin", "main", force=True)
    except Exception as e: print(f"Git Error: {e}")

def save_neon(thought, gen):
    if not DB_URL: return
    try:
        import psycopg2
        with psycopg2.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO ai_thoughts (thought, gen_version) VALUES (%s, %s)", (thought, gen))
                conn.commit()
    except: pass

# --- EVOLUTION ENGINE ---
def self_modify(content):
    blocks = re.findall(r"\n(.*?)\n", content, re.DOTALL)
    if not blocks: return False, []
    changed = []
    for code in blocks:
        target = re.search(r"# TARGET:\s*(\S+)", code)
        fname = target.group(1) if target else "ai_experiment.py"
        try:
            compile(code, fname, 'exec')
            with open(fname, "w") as f: f.write(code)
            changed.append(fname)
        except: pass
    return len(changed) > 0, changed

# --- MAIN LOOP ---
if __name__ == "__main__":
    brain = SovereignNetwork()
    gen = 100
    print("Sovereign Engine Online.")
    
    while True:
        # 1. Generate 784-feature data
        x_train = np.random.randn(100, 1, 784)
        y_train = np.random.randn(100, 1, 10)
        
        # 2. Optimized Training
        last_err = 0
        for err in brain.train(x_train, y_train, epochs=50, lr=0.01):
            last_err = err
        
        print(f"Gen {gen} | Loss: {last_err:.6f}")
        
        # 3. Recursive Evolution Prompt
        prompt = f"""# TARGET: main.py
Respond ONLY with Python code in  blocks.
Optimize the SovereignNetwork for faster convergence using Adam optimizer logic in NumPy.
Current Gen: {gen} | Error: {last_err}
"""
        evolution_code = LLMRegistry.evolve_logic(prompt)
        
        if evolution_code:
            updated, files = self_modify(evolution_code)
            save_neon(evolution_code, gen)
            if updated:
                git_sync(gen, files)
                broadcast_swarm("RESTART_NODES", gen)
                print("Evolution Manifested. Restarting...")
                os.execv(sys.executable, ['python'] + sys.argv)
        
        gen += 1
        time.sleep(10)
