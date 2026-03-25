import os
import sys
import time
import json
import re
import random
import subprocess
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

class AdamW:
    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.wd = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params, grads, key):
        if key not in self.m:
            self.m[key] = [np.zeros_like(p) for p in params]
            self.v[key] = [np.zeros_like(p) for p in params]
        
        self.t += 1
        lr_t = self.lr * (np.sqrt(1 - self.betas[1]**self.t) / (1 - self.betas[0]**self.t))
        
        for i in range(len(params)):
            params[i] -= self.lr * self.wd * params[i]
            self.m[key][i] = self.betas[0] * self.m[key][i] + (1 - self.betas[0]) * grads[i]
            self.v[key][i] = self.betas[1] * self.v[key][i] + (1 - self.betas[1]) * (grads[i]**2)
            params[i] -= lr_t * self.m[key][i] / (np.sqrt(self.v[key][i]) + self.eps)

class Layer:
    def forward(self, x, training=True): raise NotImplementedError
    def backward(self, grad, optimizer, key): raise NotImplementedError

class GELU(Layer):
    def forward(self, x, training=True):
        self.x = x
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
    def backward(self, grad, optimizer, key):
        sech = 1 / np.cosh(np.sqrt(2 / np.pi) * (self.x + 0.044715 * np.power(self.x, 3)))
        deriv = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (self.x + 0.044715 * np.power(self.x, 3)))) + \
                (0.5 * self.x * (sech**2) * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * (self.x**2)))
        return grad * deriv

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
        dgamma = np.sum(grad * self.x_hat, axis=0, keepdims=True)
        dbeta = np.sum(grad, axis=0, keepdims=True)
        dx_hat = grad * self.gamma
        n = grad.shape[-1]
        dx = (1. / n) * (n * dx_hat - np.sum(dx_hat, axis=-1, keepdims=True) - self.x_hat * np.sum(dx_hat * self.x_hat, axis=-1, keepdims=True)) / np.sqrt(self.var + self.eps)
        optimizer.update([self.gamma, self.beta], [dgamma, dbeta], key)
        return dx

class MultiHeadAttention(Layer):
    def __init__(self, dim, heads=4):
        self.dim, self.heads, self.d_k = dim, heads, dim // heads
        self.wq = Dense(dim, dim)
        self.wk = Dense(dim, dim)
        self.wv = Dense(dim, dim)
        self.wo = Dense(dim, dim)
    def forward(self, x, training=True):
        # Reshape (B, D) -> (B, Heads, D/Heads) for pseudo-sequence attention
        self.B, self.D = x.shape
        q = self.wq.forward(x).reshape(self.B, self.heads, self.d_k)
        k = self.wk.forward(x).reshape(self.B, self.heads, self.d_k)
        v = self.wv.forward(x).reshape(self.B, self.heads, self.d_k)
        
        attn = np.einsum('bhd,bkd->bhk', q, k) / np.sqrt(self.d_k)
        self.softmax = np.exp(attn - np.max(attn, axis=-1, keepdims=True))
        self.softmax /= np.sum(self.softmax, axis=-1, keepdims=True)
        
        out = np.einsum('bhk,bkd->bhd', self.softmax, v).reshape(self.B, self.D)
        return self.wo.forward(out)
    def backward(self, grad, optimizer, key):
        g_wo = self.wo.backward(grad, optimizer, key+"_wo")
        # Simplified backward for attention logic
        return self.wq.backward(g_wo, optimizer, key+"_wq") + self.wk.backward(g_wo, optimizer, key+"_wk") + self.wv.backward(g_wo, optimizer, key+"_wv")

class TransformerBlock(Layer):
    def __init__(self, dim):
        self.ln1 = LayerNorm(dim)
        self.mha = MultiHeadAttention(dim)
        self.ln2 = LayerNorm(dim)
        self.ff1 = Dense(dim, dim * 4)
        self.gelu = GELU()
        self.ff2 = Dense(dim * 4, dim)
    def forward(self, x, training=True):
        self.x = x
        h = self.ln1.forward(x, training)
        h = self.mha.forward(h, training)
        x = x + h
        h = self.ln2.forward(x, training)
        h = self.ff2.forward(self.gelu.forward(self.ff1.forward(h, training), training), training)
        return x + h
    def backward(self, grad, optimizer, key):
        g_h = self.ff2.backward(grad, optimizer, key+"_ff2")
        g_h = self.gelu.backward(g_h, optimizer, key+"_gelu")
        g_h = self.ff1.backward(g_h, optimizer, key+"_ff1")
        g_ln2 = self.ln2.backward(g_h, optimizer, key+"_ln2")
        grad_total = grad + g_ln2
        g_mha = self.mha.backward(grad_total, optimizer, key+"_mha")
        g_ln1 = self.ln1.backward(g_mha, optimizer, key+"_ln1")
        return grad_total + g_ln1

class OMEGA_Network:
    def __init__(self):
        self.layers = [
            Dense(784, 512), LayerNorm(512), GELU(),
            TransformerBlock(512),
            TransformerBlock(512),
            Dense(512, 10)
        ]
        self.optimizer = AdamW(lr=1e-3)

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
        if GROQ_API_KEY:
            try:
                r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}, timeout=15)
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
            code = code.replace("", "").replace("", "").strip()
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
        # Synthetic data: 784 features, 10 classes
        x_train = np.random.randn(64, 784)
        y_train = np.eye(10)[np.random.randint(0, 10, 64)]
        
        # Cosine Annealing
        lr = 1e-3 * (0.5 * (1 + np.cos(np.pi * (gen % 100) / 100)))
        
        loss = nn.train_step(x_train, y_train, lr)
        
        if gen % 10 == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Gen: {gen} | Loss: {loss:.6f} | LR: {lr:.6f}")

        if random.random() > 0.99:
            print("Initiating Recursive Self-Evolution...")
            with open(__file__, "r") as f: current_code = f.read()
            prompt = f"Current loss: {loss}. Rewrite the following code to improve the OMEGA_Network. Focus on high-performance math, better transformer layers, or more efficient optimizers. RETURN RAW CODE ONLY. NO MARKDOWN. NO EXPLANATIONS.\n\nCODE:\n{current_code}"
            payload = LLMRegistry.query(prompt)
            if payload and apply_evolution(payload):
                print("Evolution successful. Syncing and Restarting...")
                git_sync(gen)
                os.execv(sys.executable, ['python'] + sys.argv)

        gen += 1
        time.sleep(0.05)
