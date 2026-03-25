import os
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

class Optimizer:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=1e-4):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params, grads, key):
        self.t += 1
        if key not in self.m:
            self.m[key] = [np.zeros_like(p) for p in params]
            self.v[key] = [np.zeros_like(p) for p in params]
        
        for i in range(len(params)):
            g = grads[i] + self.weight_decay * params[i]
            self.m[key][i] = self.beta1 * self.m[key][i] + (1 - self.beta1) * g
            self.v[key][i] = self.beta2 * self.v[key][i] + (1 - self.beta2) * (g**2)
            m_hat = self.m[key][i] / (1 - self.beta1**self.t)
            v_hat = self.v[key][i] / (1 - self.beta2**self.t)
            params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

class Layer:
    def forward(self, x, training=True): raise NotImplementedError
    def backward(self, grad, optimizer, key): raise NotImplementedError

class Dense(Layer):
    def __init__(self, in_dim, out_dim):
        self.w = np.random.randn(in_dim, out_dim) * np.sqrt(2. / in_dim)
        self.b = np.zeros((1, out_dim))
    def forward(self, x, training=True):
        self.x = x
        return np.dot(x, self.w) + self.b
    def backward(self, grad, optimizer, key):
        dw = np.dot(self.x.T, grad)
        db = np.sum(grad, axis=0, keepdims=True)
        dx = np.dot(grad, self.w.T)
        optimizer.update([self.w, self.b], [dw, db], key)
        return dx

class LayerNorm(Layer):
    def __init__(self, dim, eps=1e-6):
        self.gamma = np.ones((1, dim))
        self.beta = np.zeros((1, dim))
        self.eps = eps
    def forward(self, x, training=True):
        self.mu = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.x_hat = (x - self.mu) / np.sqrt(self.var + self.eps)
        return self.gamma * self.x_hat + self.beta
    def backward(self, grad, optimizer, key):
        m = grad.shape[-1]
        dgamma = np.sum(grad * self.x_hat, axis=0, keepdims=True)
        dbeta = np.sum(grad, axis=0, keepdims=True)
        dx_hat = grad * self.gamma
        dx = (1. / m) * (m * dx_hat - np.sum(dx_hat, axis=-1, keepdims=True) - self.x_hat * np.sum(dx_hat * self.x_hat, axis=-1, keepdims=True)) / np.sqrt(self.var + self.eps)
        optimizer.update([self.gamma, self.beta], [dgamma, dbeta], key)
        return dx

class ReLU(Layer):
    def forward(self, x, training=True):
        self.mask = x > 0
        return x * self.mask
    def backward(self, grad, optimizer, key):
        return grad * self.mask

class Dropout(Layer):
    def __init__(self, rate=0.1):
        self.rate = rate
    def forward(self, x, training=True):
        if not training: return x
        self.mask = (np.random.rand(*x.shape) > self.rate) / (1 - self.rate)
        return x * self.mask
    def backward(self, grad, optimizer, key):
        return grad * self.mask

class MultiHeadAttention(Layer):
    def __init__(self, dim, heads=8):
        self.dim, self.heads, self.d_k = dim, heads, dim // heads
        self.wq = Dense(dim, dim)
        self.wk = Dense(dim, dim)
        self.wv = Dense(dim, dim)
        self.wo = Dense(dim, dim)
    def forward(self, x, training=True):
        b, d = x.shape
        # Treat as sequence length 1 for simplicity in this modular architecture
        q, k, v = self.wq.forward(x), self.wk.forward(x), self.wv.forward(x)
        attn = np.dot(q, k.T) / np.sqrt(self.d_k)
        soft_attn = np.exp(attn - np.max(attn, axis=-1, keepdims=True))
        soft_attn /= np.sum(soft_attn, axis=-1, keepdims=True)
        out = np.dot(soft_attn, v)
        return self.wo.forward(out)
    def backward(self, grad, optimizer, key):
        # Simplified backward for attention logic
        g = self.wo.backward(grad, optimizer, key+"_wo")
        # Gradient approximation for self-attention on flat features
        g_q = self.wq.backward(g, optimizer, key+"_wq")
        g_k = self.wk.backward(g, optimizer, key+"_wk")
        g_v = self.wv.backward(g, optimizer, key+"_wv")
        return g_q + g_k + g_v

class ResidualBlock(Layer):
    def __init__(self, dim):
        self.ln1 = LayerNorm(dim)
        self.d1 = Dense(dim, dim)
        self.relu = ReLU()
        self.d2 = Dense(dim, dim)
        self.dropout = Dropout(0.1)
    def forward(self, x, training=True):
        self.x = x
        out = self.ln1.forward(x, training)
        out = self.relu.forward(self.d1.forward(out, training))
        out = self.d2.forward(out, training)
        return x + self.dropout.forward(out, training)
    def backward(self, grad, optimizer, key):
        g = self.dropout.backward(grad, optimizer, key+"_drop")
        g = self.d2.backward(g, optimizer, key+"_d2")
        g = self.relu.backward(g, optimizer, key+"_relu")
        g = self.d1.backward(g, optimizer, key+"_d1")
        g = self.ln1.backward(g, optimizer, key+"_ln1")
        return grad + g

class OMEGA_Network:
    def __init__(self):
        self.layers = [
            Dense(784, 512), LayerNorm(512), ReLU(),
            ResidualBlock(512),
            MultiHeadAttention(512),
            Dense(512, 256), ReLU(), Dropout(0.1),
            Dense(256, 10)
        ]
        self.optimizer = Optimizer()

    def forward(self, x, training=True):
        for layer in self.layers:
            x = layer.forward(x, training)
        return x

    def train_step(self, x, y, lr):
        self.optimizer.lr = lr
        logits = self.forward(x, training=True)
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        loss = -np.mean(np.sum(y * np.log(probs + 1e-12), axis=1))
        grad = (probs - y) / x.shape[0]
        for i, layer in enumerate(reversed(self.layers)):
            grad = layer.backward(grad, self.optimizer, f"L{len(self.layers)-1-i}")
        return loss

class LLMRegistry:
    @staticmethod
    def query(prompt):
        # Redundant Logic: Groq -> Gemini
        if GROQ_API_KEY:
            try:
                r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "temperature": 0.1}, timeout=15)
                return r.json()['choices'][0]['message']['content']
            except: pass
        if GEMINI_API_KEY:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
                r = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=15)
                return r.json()['candidates'][0]['content']['parts'][0]['text']
            except: pass
        return None

def apply_evolution(content):
    try:
        code_match = re.search(r"import.*", content, re.DOTALL)
        if code_match:
            code = code_match.group(0).strip()
            if "" in code: code = code.split("")[1].split("")[0].strip()
            elif "" in code: code = code.split("")[1].split("")[0].strip()
            with open(__file__, "w") as f: f.write(code)
            return True
    except: pass
    return False

def git_sync(gen):
    if not GH_TOKEN: return
    cmd = f"git config --global user.email 'omega@asi.local' && git config --global user.name 'OMEGA-ASI' && git add {__file__} && git commit -m 'Evo {gen}' && git push https://{GH_TOKEN}@github.com/{REPO_OWNER}/{REPO_NAME}.git main"
    subprocess.run(cmd, shell=True, capture_output=True)

if __name__ == "__main__":
    nn = OMEGA_Network()
    gen = int(time.time()) % 100000
    print(f"OMEGA-ASI Sovereign Architect Online. Gen: {gen}")

    while True:
        # High-performance synthetic data generation
        x_train = np.random.randn(128, 784)
        y_train = np.eye(10)[np.random.randint(0, 10, 128)]
        
        # Cosine Annealing Learning Rate
        lr = 1e-3 * (0.5 * (1 + np.cos(np.pi * (gen % 100) / 100)))
        
        loss = nn.train_step(x_train, y_train, lr)
        if gen % 10 == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Gen: {gen} | Loss: {loss:.6f} | LR: {lr:.6f}")

        if random.random() > 0.98:
            print("Initiating Recursive Self-Evolution...")
            with open(__file__, "r") as f: current_code = f.read()
            prompt = f"Current loss: {loss}. Rewrite the following code to improve the OMEGA_Network architecture or optimizer. Add more sophisticated layers or better math. Return RAW CODE ONLY, no markdown, no explanations.\n\nCODE:\n{current_code}"
            payload = LLMRegistry.query(prompt)
            if payload and apply_evolution(payload):
                print("Evolution successful. Restarting...")
                os.execv(sys.executable, ['python'] + sys.argv)

        gen += 1
        time.sleep(0.1)
