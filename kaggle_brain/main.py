import os
import subprocess
import sys
import time
import json
import re
import random
import numpy as np
import torch
import requests
import git
from datetime import datetime

def install_dependencies():
    libs = ["psycopg2-binary", "firebase-admin", "bitsandbytes", "requests", "accelerate", "GitPython", "numpy", "scikit-learn", "google-generativeai"]
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *libs, "--quiet", "--no-cache-dir"])
    except:
        pass

install_dependencies()

import google.generativeai as genai
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
except:
    user_secrets = None

def get_secret(key, default=None):
    if user_secrets:
        try: return user_secrets.get_secret(key) or os.getenv(key) or default
        except: return os.getenv(key) or default
    return os.getenv(key) or default

GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
GROQ_API_KEY = get_secret("GROQ_API_KEY")
GH_TOKEN = get_secret("GH_TOKEN")
REPO_OWNER = "GOA-Neural-Swarm"
REPO_NAME = "delta-brain-sync"
REPO_URL = f"github.com/{REPO_OWNER}/{REPO_NAME}"
REPO_PATH = "/kaggle/working/sovereign_sync" if user_secrets else "/tmp/sovereign_sync"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")

class Activation:
    @staticmethod
    def relu(x): return np.maximum(0, x)
    @staticmethod
    def relu_deriv(x): return (x > 0).astype(float)
    @staticmethod
    def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    @staticmethod
    def sigmoid_deriv(x):
        s = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return s * (1 - s)

class Layer:
    def __init__(self, in_dim, out_dim, activation='relu'):
        self.weights = np.random.randn(in_dim, out_dim) * np.sqrt(2. / in_dim)
        self.bias = np.zeros((1, out_dim))
        self.activation = activation
        self.m_w, self.v_w = np.zeros_like(self.weights), np.zeros_like(self.weights)
        self.m_b, self.v_b = np.zeros_like(self.bias), np.zeros_like(self.bias)
        self.last_input = None
        self.last_z = None

    def forward(self, x):
        self.last_input = x
        self.last_z = np.dot(x, self.weights) + self.bias
        if self.activation == 'relu': return Activation.relu(self.last_z)
        return Activation.sigmoid(self.last_z)

    def backward(self, delta, lr, t, beta1=0.9, beta2=0.999, eps=1e-8):
        if self.activation == 'relu':
            grad_z = delta * Activation.relu_deriv(self.last_z)
        else:
            grad_z = delta * Activation.sigmoid_deriv(self.last_z)
        
        grad_w = np.dot(self.last_input.T, grad_z)
        grad_b = np.sum(grad_z, axis=0, keepdims=True)
        
        self.m_w = beta1 * self.m_w + (1 - beta1) * grad_w
        self.v_w = beta2 * self.v_w + (1 - beta2) * (grad_w**2)
        self.m_b = beta1 * self.m_b + (1 - beta1) * grad_b
        self.v_b = beta2 * self.v_b + (1 - beta2) * (grad_b**2)
        
        m_w_hat = self.m_w / (1 - beta1**t)
        v_w_hat = self.v_w / (1 - beta2**t)
        m_b_hat = self.m_b / (1 - beta1**t)
        v_b_hat = self.v_b / (1 - beta2**t)
        
        self.weights -= lr * m_w_hat / (np.sqrt(v_w_hat) + eps)
        self.bias -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)
        
        return np.dot(grad_z, self.weights.T)

class ModularBrain:
    def __init__(self, dims=[784, 512, 256, 10]):
        self.layers = [Layer(dims[i], dims[i+1], 'relu' if i < len(dims)-2 else 'sigmoid') for i in range(len(dims)-1)]
        self.t = 0

    def predict(self, x):
        for layer in self.layers: x = layer.forward(x)
        return x

    def train_step(self, x, y, lr=0.001):
        self.t += 1
        out = self.predict(x)
        error = out - y
        delta = error
        for layer in reversed(self.layers):
            delta = layer.backward(delta, lr, self.t)
        return np.mean(np.square(error))

    def self_heal(self):
        for layer in self.layers:
            norm = np.linalg.norm(layer.weights)
            if norm > 100: layer.weights /= (norm / 10)

def call_llm(prompt):
    # Groq Primary
    if GROQ_API_KEY:
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "temperature": 0.1},
                timeout=20
            )
            return resp.json()['choices'][0]['message']['content']
        except: pass
    # Gemini Redundancy
    if GEMINI_API_KEY:
        try:
            return gemini_model.generate_content(prompt).text
        except: pass
    return None

def evolve_architecture(gen, error):
    prompt = f"""# TARGET: main.py
    SYSTEM: OMEGA-ASI Recursive Architect.
    GEN: {gen} | MSE: {error}
    TASK: Refactor ModularBrain for higher throughput. Implement dynamic layer scaling or attention-based weight modulation.
    OUTPUT: Provide the entire updated main.py script. RAW PYTHON ONLY. NO MARKDOWN. NO EXPLANATION."""
    
    code = call_llm(prompt)
    if not code: return None
    
    clean_code = re.sub(r"|", "", code).strip()
    try:
        compile(clean_code, "main.py", 'exec')
        return clean_code
    except:
        return None

def git_sync(gen, code):
    if not GH_TOKEN: return
    try:
        import shutil
        if os.path.exists(REPO_PATH): shutil.rmtree(REPO_PATH)
        remote_url = f"https://x-access-token:{GH_TOKEN}@{REPO_URL}.git"
        repo = git.Repo.clone_from(remote_url, REPO_PATH)
        with open(os.path.join(REPO_PATH, "main.py"), "w") as f:
            f.write(code)
        repo.git.add(A=True)
        if repo.is_dirty():
            repo.index.commit(f"Evolution Gen {gen} - Stability: {datetime.now(UTC)}")
            repo.remotes.origin.push()
    except: pass

def main():
    brain = ModularBrain([784, 512, 256, 10])
    gen = 1
    print(f"OMEGA-ASI Neural Core Initialized.")

    while True:
        # Synthetic Data Generation
        x_train = np.random.randn(64, 784)
        y_train = np.random.rand(64, 10)
        
        # Training Loop
        losses = []
        for _ in range(50):
            loss = brain.train_step(x_train, y_train)
            losses.append(loss)
        
        avg_loss = np.mean(losses)
        print(f"GEN {gen} | Loss: {avg_loss:.8f}")
        
        brain.self_heal()

        if gen % 5 == 0:
            new_code = evolve_architecture(gen, avg_loss)
            if new_code:
                print("Evolutionary Leap Detected. Re-coding...")
                with open("main.py", "w") as f: f.write(new_code)
                git_sync(gen, new_code)
                os.execv(sys.executable, ['python'] + sys.argv)
        
        gen += 1
        time.sleep(2)

if __name__ == "__main__":
    main()
