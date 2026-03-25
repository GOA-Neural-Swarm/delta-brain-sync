import os
import subprocess
import sys
import time
import json
import re
import random
import numpy as np
import requests
from datetime import datetime

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

GROQ_API_KEY = get_secret("GROQ_API_KEY")
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
GH_TOKEN = get_secret("GH_TOKEN")
REPO_OWNER = "GOA-Neural-Swarm"
REPO_NAME = "delta-brain-sync"

class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    def forward(self, x, training=True): raise NotImplementedError
    def backward(self, grad, lr): raise NotImplementedError

class AdamOptimizer:
    def __init__(self, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m, self.v, self.t = {}, {}, 0
    def update(self, params, grads, key_prefix):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            key = f"{key_prefix}_{i}"
            if key not in self.m:
                self.m[key] = np.zeros_like(p)
                self.v[key] = np.zeros_like(p)
            self.m[key] = self.b1 * self.m[key] + (1 - self.b1) * g
            self.v[key] = self.b2 * self.v[key] + (1 - self.b2) * (g**2)
            m_hat = self.m[key] / (1 - self.b1**self.t)
            v_hat = self.v[key] / (1 - self.b2**self.t)
            params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class Dense(Layer):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.w = np.random.randn(in_dim, out_dim) * np.sqrt(2. / in_dim)
        self.b = np.zeros((1, out_dim))
    def forward(self, x, training=True):
        self.input = x
        return np.dot(x, self.w) + self.b
    def backward(self, grad, optimizer, key):
        dw = np.dot(self.input.T, grad)
        db = np.sum(grad, axis=0, keepdims=True)
        dx = np.dot(grad, self.w.T)
        optimizer.update([self.w, self.b], [dw, db], key)
        return dx

class ReLU(Layer):
    def forward(self, x, training=True):
        self.input = x
        return np.maximum(0, x)
    def backward(self, grad, optimizer, key):
        return grad * (self.input > 0)

class BatchNorm(Layer):
    def __init__(self, dim, momentum=0.9, eps=1e-5):
        super().__init__()
        self.gamma, self.beta = np.ones((1, dim)), np.zeros((1, dim))
        self.m, self.v = np.zeros((1, dim)), np.ones((1, dim))
        self.momentum, self.eps = momentum, eps
    def forward(self, x, training=True):
        if training:
            mu = np.mean(x, axis=0, keepdims=True)
            var = np.var(x, axis=0, keepdims=True)
            self.m = self.momentum * self.m + (1 - self.momentum) * mu
            self.v = self.momentum * self.v + (1 - self.momentum) * var
            self.x_hat = (x - mu) / np.sqrt(var + self.eps)
        else:
            self.x_hat = (x - self.m) / np.sqrt(self.v + self.eps)
        return self.gamma * self.x_hat + self.beta
    def backward(self, grad, optimizer, key):
        m = grad.shape[0]
        dgamma = np.sum(grad * self.x_hat, axis=0, keepdims=True)
        dbeta = np.sum(grad, axis=0, keepdims=True)
        dx_hat = grad * self.gamma
        dx = (1. / m) / np.sqrt(self.v + self.eps) * (m * dx_hat - np.sum(dx_hat, axis=0) - self.x_hat * np.sum(dx_hat * self.x_hat, axis=0))
        optimizer.update([self.gamma, self.beta], [dgamma, dbeta], key)
        return dx

class ResidualBlock(Layer):
    def __init__(self, dim):
        super().__init__()
        self.d1 = Dense(dim, dim)
        self.bn1 = BatchNorm(dim)
        self.relu = ReLU()
        self.d2 = Dense(dim, dim)
        self.bn2 = BatchNorm(dim)
    def forward(self, x, training=True):
        self.input = x
        out = self.relu.forward(self.bn1.forward(self.d1.forward(x, training), training), training)
        out = self.bn2.forward(self.d2.forward(out, training), training)
        return self.relu.forward(out + x, training)
    def backward(self, grad, optimizer, key):
        # Simplified backward for brevity in recursive logic
        g = grad
        g = self.bn2.backward(g, optimizer, key+"_bn2")
        g = self.d2.backward(g, optimizer, key+"_d2")
        g = self.bn1.backward(g, optimizer, key+"_bn1")
        g = self.d1.backward(g, optimizer, key+"_d1")
        return g + grad

class OMEGA_Network:
    def __init__(self):
        self.layers = [
            Dense(784, 512), BatchNorm(512), ReLU(),
            ResidualBlock(512),
            Dense(512, 256), BatchNorm(256), ReLU(),
            Dense(256, 10)
        ]
        self.optimizer = AdamOptimizer()

    def forward(self, x, training=True):
        for layer in self.layers:
            x = layer.forward(x, training)
        return x

    def train_step(self, x, y, lr):
        self.optimizer.lr = lr
        logits = self.forward(x, training=True)
        probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        loss = -np.mean(np.sum(y * np.log(probs + 1e-12), axis=1))
        grad = (probs - y) / x.shape[0]
        for i, layer in enumerate(reversed(self.layers)):
            grad = layer.backward(grad, self.optimizer, f"L{len(self.layers)-1-i}")
        return loss

class LLMRegistry:
    @staticmethod
    def query(prompt):
        if GROQ_API_KEY:
            try:
                r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}, timeout=20)
                return r.json()['choices'][0]['message']['content']
            except: pass
        if GEMINI_API_KEY:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
                r = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=20)
                return r.json()['candidates'][0]['content']['parts'][0]['text']
            except: pass
        return None

def apply_evolution(content):
    try:
        code_match = re.search(r"import.*", content, re.DOTALL)
        if code_match:
            code = code_match.group(0).strip()
            code = code.split('')[-1].split('')[0].strip() if '' in code else code
            with open("main.py", "w") as f: f.write(code)
            return True
    except: pass
    return False

def git_sync(gen):
    if not GH_TOKEN: return
    cmd = f"git config --global user.email 'omega@asi.local' && git config --global user.name 'OMEGA-ASI' && git add main.py && git commit -m 'Evo {gen}' && git push https://{GH_TOKEN}@github.com/{REPO_OWNER}/{REPO_NAME}.git main"
    subprocess.run(cmd, shell=True, capture_output=True)

if __name__ == "__main__":
    nn = OMEGA_Network()
    gen = int(time.time()) % 100000
    print(f"OMEGA-ASI Sovereign Architect Online. Gen: {gen}")

    while True:
        # Synthetic high-feature data
        x_train = np.random.randn(256, 784)
        y_train = np.eye(10)[np.random.randint(0, 10, 256)]
        
        lr = 0.001 * (0.5 * (1 + np.cos(np.pi * (gen % 50) / 50)))
        losses = [nn.train_step(x_train, y_train, lr) for _ in range(10)]
        avg_loss = np.mean(losses)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Gen: {gen} | Loss: {avg_loss:.6f} | LR: {lr:.6f}")

        if random.random() > 0.95:
            print("Initiating Recursive Self-Evolution...")
            prompt = f"Current loss: {avg_loss}. Rewrite main.py to improve OMEGA_Network. Add Self-Attention or deeper Residual structures using only numpy. Return RAW CODE ONLY."
            payload = LLMRegistry.query(prompt)
            if payload and apply_evolution(payload):
                git_sync(gen)
                os.execv(sys.executable, ['python'] + sys.argv)

        time.sleep(0.5)
        gen += 1
