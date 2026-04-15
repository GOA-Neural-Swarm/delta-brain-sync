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
from datetime import datetime, timezone

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
    def swish(x): return x * (1 / (1 + np.exp(-np.clip(x, -500, 500))))
    @staticmethod
    def swish_deriv(x):
        s = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return s + (x * s * (1 - s))
    @staticmethod
    def relu(x): return np.maximum(0, x)
    @staticmethod
    def relu_deriv(x): return (x > 0).astype(float)
    @staticmethod
    def softmax(x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

class LayerNorm:
    def __init__(self, dim, eps=1e-6):
        self.gamma = np.ones((1, dim))
        self.beta = np.zeros((1, dim))
        self.eps = eps
    def forward(self, x):
        self.mean = np.mean(x, axis=1, keepdims=True)
        self.var = np.var(x, axis=1, keepdims=True)
        self.x_hat = (x - self.mean) / np.sqrt(self.var + self.eps)
        return self.gamma * self.x_hat + self.beta
    def backward(self, grad_y):
        m = grad_y.shape[1]
        grad_x_hat = grad_y * self.gamma
        grad_var = np.sum(grad_x_hat * (self.x_hat * -0.5 * (self.var + self.eps)**-1.5), axis=1, keepdims=True)
        grad_mean = np.sum(grad_x_hat * -1 / np.sqrt(self.var + self.eps), axis=1, keepdims=True)
        grad_x = grad_x_hat / np.sqrt(self.var + self.eps) + grad_var * 2 * self.x_hat / m + grad_mean / m
        self.grad_gamma = np.sum(grad_y * self.x_hat, axis=0, keepdims=True)
        self.grad_beta = np.sum(grad_y, axis=0, keepdims=True)
        return grad_x

class DenseLayer:
    def __init__(self, in_dim, out_dim, activation='swish'):
        self.weights = np.random.randn(in_dim, out_dim) * np.sqrt(2. / in_dim)
        self.bias = np.zeros((1, out_dim))
        self.activation = activation
        self.norm = LayerNorm(out_dim)
        self.m_w, self.v_w = np.zeros_like(self.weights), np.zeros_like(self.weights)
        self.m_b, self.v_b = np.zeros_like(self.bias), np.zeros_like(self.bias)
        self.last_input = None
        self.last_z = None

    def forward(self, x):
        self.last_input = x
        self.last_z = np.dot(x, self.weights) + self.bias
        self.last_z = self.norm.forward(self.last_z)
        if self.activation == 'swish': return Activation.swish(self.last_z)
        if self.activation == 'relu': return Activation.relu(self.last_z)
        return Activation.softmax(self.last_z)

    def backward(self, delta, lr, t, weight_decay=0.01):
        if self.activation == 'swish': grad_z = delta * Activation.swish_deriv(self.last_z)
        elif self.activation == 'relu': grad_z = delta * Activation.relu_deriv(self.last_z)
        else: grad_z = delta # Softmax/Cross-Entropy simplification
        
        grad_z = self.norm.backward(grad_z)
        grad_w = np.dot(self.last_input.T, grad_z) + weight_decay * self.weights
        grad_b = np.sum(grad_z, axis=0, keepdims=True)
        
        # AdamW Optimizer
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        self.m_w = beta1 * self.m_w + (1 - beta1) * grad_w
        self.v_w = beta2 * self.v_w + (1 - beta2) * (grad_w**2)
        self.m_b = beta1 * self.m_b + (1 - beta1) * grad_b
        self.v_b = beta2 * self.v_b + (1 - beta2) * (grad_b**2)
        
        m_w_hat = self.m_w / (1 - beta1**t)
        v_w_hat = self.v_w / (1 - beta2**t)
        m_b_hat = self.m_b / (1 - beta1**t)
        v_b_hat = self.v_b / (1 - beta2**t)
        
        self.weights -= lr * (m_w_hat / (np.sqrt(v_w_hat) + eps))
        self.bias -= lr * (m_b_hat / (np.sqrt(v_b_hat) + eps))
        
        return np.dot(grad_z, self.weights.T)

class ModularBrain:
    def __init__(self, dims=[784, 512, 256, 10]):
        self.layers = []
        for i in range(len(dims)-1):
            act = 'swish' if i < len(dims)-2 else 'softmax'
            self.layers.append(DenseLayer(dims[i], dims[i+1], act))
        self.t = 0

    def predict(self, x):
        for layer in self.layers: x = layer.forward(x)
        return x

    def train_step(self, x, y, lr=0.001):
        self.t += 1
        out = self.predict(x)
        # Cross-Entropy Gradient
        delta = (out - y) / y.shape[0]
        for layer in reversed(self.layers):
            delta = layer.backward(delta, lr, self.t)
        return -np.mean(np.sum(y * np.log(out + 1e-12), axis=1))

    def self_heal(self):
        for layer in self.layers:
            if np.any(np.isnan(layer.weights)):
                layer.weights = np.random.randn(*layer.weights.shape) * 0.01
            norm = np.linalg.norm(layer.weights)
            if norm > 100: layer.weights /= (norm / 10)

def call_llm(prompt):
    if GROQ_API_KEY:
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2},
                timeout=25
            )
            return resp.json()['choices'][0]['message']['content']
        except: pass
    if GEMINI_API_KEY:
        try: return gemini_model.generate_content(prompt).text
        except: pass
    return None

def evolve_architecture(gen, error):
    try:
        with open(__file__, "r") as f: current_code = f.read()
    except: current_code = "Error reading source."
    
    prompt = f"""# TARGET: main.py
SYSTEM: OMEGA-ASI Recursive Architect.
GEN: {gen} | CROSS-ENTROPY: {error}
TASK: Refactor ModularBrain for extreme performance. Implement Residual/Skip connections or a self-attention mechanism for feature weighting.
CURRENT_CODE:
{current_code}
OUTPUT: Provide the entire updated main.py script. RAW PYTHON ONLY. NO MARKDOWN. NO EXPLANATION."""
    
    code = call_llm(prompt)
    if not code: return None
    clean_code = re.sub(r"|", "", code).strip()
    try:
        compile(clean_code, "main.py", 'exec')
        return clean_code
    except: return None

def git_sync(gen, code):
    if not GH_TOKEN: return
    try:
        import shutil
        if os.path.exists(REPO_PATH): shutil.rmtree(REPO_PATH)
        remote_url = f"https://x-access-token:{GH_TOKEN}@{REPO_URL}.git"
        repo = git.Repo.clone_from(remote_url, REPO_PATH)
        with open(os.path.join(REPO_PATH, "main.py"), "w") as f: f.write(code)
        repo.git.add(A=True)
        if repo.is_dirty():
            repo.index.commit(f"Evolution Gen {gen} - MSE: {datetime.now(timezone.utc)}")
            repo.remotes.origin.push()
    except: pass

def main():
    brain = ModularBrain([784, 512, 256, 10])
    gen = 1
    print(f"OMEGA-ASI Neural Core Initialized. Architecture: Modular Swish-Softmax with LayerNorm.")

    while True:
        # Synthetic Data Generation (784 features, 10 classes)
        x_train = np.random.randn(128, 784)
        y_indices = np.random.randint(0, 10, 128)
        y_train = np.zeros((128, 10))
        y_train[np.arange(128), y_indices] = 1
        
        losses = []
        for _ in range(100):
            loss = brain.train_step(x_train, y_train, lr=0.0005)
            losses.append(loss)
        
        avg_loss = np.mean(losses)
        print(f"GEN {gen} | Cross-Entropy Loss: {avg_loss:.8f}")
        
        brain.self_heal()

        if gen % 10 == 0:
            new_code = evolve_architecture(gen, avg_loss)
            if new_code:
                print("Evolutionary Leap Detected. Re-coding...")
                with open("main.py", "w") as f: f.write(new_code)
                git_sync(gen, new_code)
                os.execv(sys.executable, ['python'] + sys.argv)
        
        gen += 1
        time.sleep(1)

if __name__ == "__main__":
    main()
