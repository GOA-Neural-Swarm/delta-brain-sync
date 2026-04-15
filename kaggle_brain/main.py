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
    def softmax(x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

class LayerNorm:
    def __init__(self, dim, eps=1e-6):
        self.gamma = np.ones((1, dim))
        self.beta = np.zeros((1, dim))
        self.eps = eps
    def forward(self, x):
        self.x = x
        self.mean = np.mean(x, axis=1, keepdims=True)
        self.var = np.var(x, axis=1, keepdims=True)
        self.x_hat = (x - self.mean) / np.sqrt(self.var + self.eps)
        return self.gamma * self.x_hat + self.beta
    def backward(self, grad_y):
        m = grad_y.shape[1]
        grad_x_hat = grad_y * self.gamma
        grad_var = np.sum(grad_x_hat * (self.x - self.mean) * -0.5 * (self.var + self.eps)**-1.5, axis=1, keepdims=True)
        grad_mean = np.sum(grad_x_hat * -1 / np.sqrt(self.var + self.eps), axis=1, keepdims=True) + grad_var * np.mean(-2 * (self.x - self.mean), axis=1, keepdims=True)
        grad_x = grad_x_hat / np.sqrt(self.var + self.eps) + grad_var * 2 * (self.x - self.mean) / m + grad_mean / m
        self.grad_gamma = np.sum(grad_y * self.x_hat, axis=0, keepdims=True)
        self.grad_beta = np.sum(grad_y, axis=0, keepdims=True)
        return grad_x

class AttentionLayer:
    def __init__(self, dim):
        self.dim = dim
        self.wq = np.random.randn(dim, dim) * np.sqrt(2./dim)
        self.wk = np.random.randn(dim, dim) * np.sqrt(2./dim)
        self.wv = np.random.randn(dim, dim) * np.sqrt(2./dim)
        self.m_q, self.v_q = np.zeros_like(self.wq), np.zeros_like(self.wq)
        self.m_k, self.v_k = np.zeros_like(self.wk), np.zeros_like(self.wk)
        self.m_v, self.v_v = np.zeros_like(self.wv), np.zeros_like(self.wv)

    def forward(self, x):
        self.last_x = x
        self.q = np.dot(x, self.wq)
        self.k = np.dot(x, self.wk)
        self.v = np.dot(x, self.wv)
        self.scores = np.dot(self.q, self.k.T) / np.sqrt(self.dim)
        self.probs = Activation.softmax(self.scores)
        return np.dot(self.probs, self.v)

    def backward(self, grad_out, lr, t):
        grad_v = np.dot(self.probs.T, grad_out)
        grad_probs = np.dot(grad_out, self.v.T)
        # Simplified softmax backward
        grad_scores = self.probs * (grad_probs - np.sum(grad_probs * self.probs, axis=1, keepdims=True))
        grad_q = np.dot(grad_scores, self.k) / np.sqrt(self.dim)
        grad_k = np.dot(grad_scores.T, self.q) / np.sqrt(self.dim)
        
        grad_wq = np.dot(self.last_x.T, grad_q)
        grad_wk = np.dot(self.last_x.T, grad_k)
        grad_wv = np.dot(self.last_x.T, grad_v)

        for param, grad, m, v in zip([self.wq, self.wk, self.wv], [grad_wq, grad_wk, grad_wv], [self.m_q, self.m_k, self.m_v], [self.v_q, self.v_k, self.v_v]):
            m[:] = 0.9 * m + 0.1 * grad
            v[:] = 0.999 * v + 0.001 * (grad**2)
            m_hat = m / (1 - 0.9**t)
            v_hat = v / (1 - 0.999**t)
            param -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        
        return np.dot(grad_q, self.wq.T) + np.dot(grad_k, self.wk.T) + np.dot(grad_v, self.wv.T)

class ResidualBlock:
    def __init__(self, dim):
        self.w1 = np.random.randn(dim, dim) * np.sqrt(2./dim)
        self.b1 = np.zeros((1, dim))
        self.norm1 = LayerNorm(dim)
        self.w2 = np.random.randn(dim, dim) * np.sqrt(2./dim)
        self.b2 = np.zeros((1, dim))
        self.norm2 = LayerNorm(dim)
        self.m_w1, self.v_w1 = np.zeros_like(self.w1), np.zeros_like(self.w1)
        self.m_b1, self.v_b1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.m_w2, self.v_w2 = np.zeros_like(self.w2), np.zeros_like(self.w2)
        self.m_b2, self.v_b2 = np.zeros_like(self.b2), np.zeros_like(self.b2)

    def forward(self, x):
        self.last_x = x
        z1 = self.norm1.forward(np.dot(x, self.w1) + self.b1)
        self.a1 = Activation.swish(z1)
        z2 = self.norm2.forward(np.dot(self.a1, self.w2) + self.b2)
        return self.a1 + x # Residual connection

    def backward(self, grad_out, lr, t):
        grad_a1 = grad_out # from residual
        grad_z2 = grad_out # simplified
        grad_z2 = self.norm2.backward(grad_z2)
        grad_w2 = np.dot(self.a1.T, grad_z2)
        grad_b2 = np.sum(grad_z2, axis=0, keepdims=True)
        grad_a1 += np.dot(grad_z2, self.w2.T)
        
        grad_z1 = grad_a1 * Activation.swish_deriv(self.a1)
        grad_z1 = self.norm1.backward(grad_z1)
        grad_w1 = np.dot(self.last_x.T, grad_z1)
        grad_b1 = np.sum(grad_z1, axis=0, keepdims=True)
        
        grad_x = np.dot(grad_z1, self.w1.T) + grad_out

        for p, g, m, v in [(self.w1, grad_w1, self.m_w1, self.v_w1), (self.b1, grad_b1, self.m_b1, self.v_b1),
                           (self.w2, grad_w2, self.m_w2, self.v_w2), (self.b2, grad_b2, self.m_b2, self.v_b2)]:
            m[:] = 0.9 * m + 0.1 * g
            v[:] = 0.999 * v + 0.001 * (g**2)
            m_hat = m / (1 - 0.9**t)
            v_hat = v / (1 - 0.999**t)
            p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        return grad_x

class ModularBrain:
    def __init__(self, in_dim=784, hidden=512, out_dim=10):
        self.input_proj = np.random.randn(in_dim, hidden) * np.sqrt(2./in_dim)
        self.blocks = [ResidualBlock(hidden) for _ in range(3)]
        self.attention = AttentionLayer(hidden)
        self.output_proj = np.random.randn(hidden, out_dim) * np.sqrt(2./hidden)
        self.t = 0

    def predict(self, x):
        self.last_x = x
        self.z_in = np.dot(x, self.input_proj)
        curr = self.z_in
        for block in self.blocks: curr = block.forward(curr)
        self.attn_out = self.attention.forward(curr)
        self.final_z = np.dot(self.attn_out, self.output_proj)
        return Activation.softmax(self.final_z)

    def train_step(self, x, y, lr=0.001):
        self.t += 1
        probs = self.predict(x)
        delta = (probs - y) / y.shape[0]
        
        grad_out_proj = np.dot(self.attn_out.T, delta)
        grad_attn = np.dot(delta, self.output_proj.T)
        self.output_proj -= lr * grad_out_proj
        
        grad_curr = self.attention.backward(grad_attn, lr, self.t)
        for block in reversed(self.blocks):
            grad_curr = block.backward(grad_curr, lr, self.t)
        
        grad_in_proj = np.dot(self.last_x.T, grad_curr)
        self.input_proj -= lr * grad_in_proj
        
        return -np.mean(np.sum(y * np.log(probs + 1e-12), axis=1))

def call_llm(prompt):
    if GROQ_API_KEY:
        try:
            resp = requests.post("https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2},
                timeout=20)
            return resp.json()['choices'][0]['message']['content']
        except: pass
    if GEMINI_API_KEY:
        try: return gemini_model.generate_content(prompt).text
        except: pass
    return None

def evolve_architecture(gen, error):
    try:
        with open(__file__, "r") as f: current_code = f.read()
    except: current_code = ""
    prompt = f"# TARGET: main.py\nSYSTEM: OMEGA-ASI Recursive Architect.\nGEN: {gen} | LOSS: {error}\nTASK: Enhance ModularBrain with dynamic routing or sparse kernels. Optimize NumPy vectorization.\nCODE:\n{current_code}\nOUTPUT: RAW PYTHON ONLY."
    code = call_llm(prompt)
    if code:
        clean = re.sub(r"|", "", code).strip()
        try:
            compile(clean, "main.py", 'exec')
            return clean
        except: return None
    return None

def main():
    brain = ModularBrain()
    gen = 1
    print("OMEGA-ASI Sovereign Core Active. Architecture: Residual-Attention Hybrid.")
    
    while True:
        x_train = np.random.randn(64, 784)
        y_idx = np.random.randint(0, 10, 64)
        y_train = np.zeros((64, 10))
        y_train[np.arange(64), y_idx] = 1
        
        loss = brain.train_step(x_train, y_train, lr=0.0005)
        if gen % 50 == 0:
            print(f"GEN {gen} | Loss: {loss:.6f}")
            
        if gen % 500 == 0:
            new_code = evolve_architecture(gen, loss)
            if new_code:
                with open("main.py", "w") as f: f.write(new_code)
                os.execv(sys.executable, ['python'] + sys.argv)
        gen += 1

if __name__ == "__main__":
    main()
