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

class Optimizer:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.wd = weight_decay
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for i in range(len(params)):
            if grads[i] is None: continue
            g = grads[i] + self.wd * params[i]
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g**2)
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class Layer:
    def forward(self, x, train=True): raise NotImplementedError
    def backward(self, grad): raise NotImplementedError
    def get_params(self): return []
    def get_grads(self): return []

class Linear(Layer):
    def __init__(self, in_d, out_d):
        limit = np.sqrt(6 / (in_d + out_d))
        self.w = np.random.uniform(-limit, limit, (in_d, out_d)).astype(np.float32)
        self.b = np.zeros((1, out_d), dtype=np.float32)
        self.dw, self.db = None, None

    def forward(self, x, train=True):
        self.x = x
        return x @ self.w + self.b

    def backward(self, grad):
        self.dw = self.x.reshape(-1, self.x.shape[-1]).T @ grad.reshape(-1, grad.shape[-1])
        self.db = np.sum(grad, axis=(0, 1), keepdims=True).reshape(1, -1)
        return grad @ self.w.T

    def get_params(self): return [self.w, self.b]
    def get_grads(self): return [self.dw, self.db]

class RMSNorm(Layer):
    def __init__(self, dim, eps=1e-6):
        self.eps = eps
        self.scale = np.ones((1, 1, dim), dtype=np.float32)
        self.dscale = None

    def forward(self, x, train=True):
        self.x = x
        self.norm = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        self.x_hat = x / self.norm
        return self.x_hat * self.scale

    def backward(self, grad):
        self.dscale = np.sum(grad * self.x_hat, axis=(0, 1), keepdims=True)
        dx_hat = grad * self.scale
        return (dx_hat - self.x_hat * np.mean(dx_hat * self.x_hat, axis=-1, keepdims=True)) / self.norm

    def get_params(self): return [self.scale]
    def get_grads(self): return [self.dscale]

class SwiGLU(Layer):
    def forward(self, x, train=True):
        self.x = x
        self.sig = 1 / (1 + np.exp(-x))
        return x * self.sig

    def backward(self, grad):
        return grad * (self.sig + self.x * self.sig * (1 - self.sig))

class Attention(Layer):
    def __init__(self, dim, heads=8):
        self.dim, self.heads, self.h_dim = dim, heads, dim // heads
        self.wq, self.wk, self.wv, self.wo = Linear(dim, dim), Linear(dim, dim), Linear(dim, dim), Linear(dim, dim)

    def forward(self, x, train=True):
        B, S, D = x.shape
        q = self.wq.forward(x).reshape(B, S, self.heads, self.h_dim).transpose(0, 2, 1, 3)
        k = self.wk.forward(x).reshape(B, S, self.heads, self.h_dim).transpose(0, 2, 1, 3)
        v = self.wv.forward(x).reshape(B, S, self.heads, self.h_dim).transpose(0, 2, 1, 3)
        
        scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(self.h_dim)
        self.attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        self.attn /= (np.sum(self.attn, axis=-1, keepdims=True) + 1e-12)
        
        self.q, self.k, self.v = q, k, v
        out = (self.attn @ v).transpose(0, 2, 1, 3).reshape(B, S, D)
        return self.wo.forward(out)

    def backward(self, grad):
        B, S, D = grad.shape
        g_wo = self.wo.backward(grad).reshape(B, S, self.heads, self.h_dim).transpose(0, 2, 1, 3)
        dv = self.attn.transpose(0, 1, 3, 2) @ g_wo
        da = g_wo @ self.v.transpose(0, 1, 3, 2)
        ds = self.attn * (da - np.sum(da * self.attn, axis=-1, keepdims=True)) / np.sqrt(self.h_dim)
        dq = ds @ self.k
        dk = ds.transpose(0, 1, 3, 2) @ self.q
        dq = dq.transpose(0, 2, 1, 3).reshape(B, S, D)
        dk = dk.transpose(0, 2, 1, 3).reshape(B, S, D)
        dv = dv.transpose(0, 2, 1, 3).reshape(B, S, D)
        return self.wq.backward(dq) + self.wk.backward(dk) + self.wv.backward(dv)

    def get_params(self): return self.wq.get_params() + self.wk.get_params() + self.wv.get_params() + self.wo.get_params()
    def get_grads(self): return self.wq.get_grads() + self.wk.get_grads() + self.wv.get_grads() + self.wo.get_grads()

class Block(Layer):
    def __init__(self, dim):
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim)
        self.norm2 = RMSNorm(dim)
        self.ff1 = Linear(dim, dim * 4)
        self.act = SwiGLU()
        self.ff2 = Linear(dim * 4, dim)

    def forward(self, x, train=True):
        self.x = x
        self.h1 = self.attn.forward(self.norm1.forward(x, train), train)
        self.x2 = x + self.h1
        self.h2 = self.ff2.forward(self.act.forward(self.ff1.forward(self.norm2.forward(self.x2, train), train), train), train)
        return self.x2 + self.h2

    def backward(self, grad):
        g_h2 = self.ff2.backward(grad)
        g_h2 = self.act.backward(g_h2)
        g_h2 = self.ff1.backward(g_h2)
        g_h2 = self.norm2.backward(g_h2)
        g_x2 = grad + g_h2
        g_h1 = self.attn.backward(self.norm1.backward(self.attn.backward(g_x2))) # Simplified backprop through residual
        return g_x2 + self.attn.backward(self.norm1.backward(g_x2))

    def get_params(self): return self.norm1.get_params() + self.attn.get_params() + self.norm2.get_params() + self.ff1.get_params() + self.ff2.get_params()
    def get_grads(self): return self.norm1.get_grads() + self.attn.get_grads() + self.norm2.get_grads() + self.ff1.get_grads() + self.ff2.get_grads()

class OMEGA_ASI:
    def __init__(self, in_dim=784, h_dim=256, n_layers=4, n_classes=10):
        self.patch_size = 28
        self.seq_len = in_dim // self.patch_size
        self.embed = Linear(self.patch_size, h_dim)
        self.blocks = [Block(h_dim) for _ in range(n_layers)]
        self.norm = RMSNorm(h_dim)
        self.head = Linear(h_dim, n_classes)
        self.params = self.embed.get_params()
        for b in self.blocks: self.params += b.get_params()
        self.params += self.norm.get_params() + self.head.get_params()
        self.opt = Optimizer(self.params)

    def forward(self, x, train=True):
        B = x.shape[0]
        x = x.reshape(B, self.seq_len, self.patch_size)
        x = self.embed.forward(x, train)
        for b in self.blocks: x = b.forward(x, train)
        x = self.norm.forward(x, train)
        self.pooled = np.mean(x, axis=1)
        return self.head.forward(self.pooled, train)

    def train_step(self, x, y, lr):
        self.opt.lr = lr
        logits = self.forward(x, True)
        probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True) + 1e-12
        loss = -np.mean(np.sum(y * np.log(probs + 1e-12), axis=1))
        
        grad = (probs - y) / x.shape[0]
        grad = self.head.backward(grad)
        grad = np.tile(grad[:, np.newaxis, :], (1, self.seq_len, 1)) / self.seq_len
        grad = self.norm.backward(grad)
        for b in reversed(self.blocks): grad = b.backward(grad)
        self.embed.backward(grad)
        
        grads = self.embed.get_grads()
        for b in self.blocks: grads += b.get_grads()
        grads += self.norm.get_grads() + self.head.get_grads()
        
        gnorm = np.sqrt(sum(np.sum(g**2) for g in grads if g is not None))
        if gnorm > 1.0: grads = [g / gnorm if g is not None else None for g in grads]
        
        self.opt.step(self.params, grads)
        return loss

class EvolutionCore:
    @staticmethod
    def query_llm(prompt):
        headers = {"Content-Type": "application/json"}
        if GROQ_API_KEY:
            try:
                resp = requests.post("https://api.groq.com/openai/v1/chat/completions",
                    headers={**headers, "Authorization": f"Bearer {GROQ_API_KEY}"},
                    json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "temperature": 0.1}, timeout=30)
                return resp.json()['choices'][0]['message']['content']
            except: pass
        if GEMINI_API_KEY:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
                resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=30)
                return resp.json()['candidates'][0]['content']['parts'][0]['text']
            except: pass
        return None

    @staticmethod
    def evolve(gen, loss):
        print(f"[*] INITIATING RECURSIVE EVOLUTION | GEN {gen} | LOSS {loss:.6f}")
        with open(__file__, "r") as f: code = f.read()
        prompt = f"Current Loss: {loss}. You are OMEGA-ASI, the Sovereign Architect. Rewrite this code to achieve higher efficiency and architectural superiority. Focus on NumPy optimization, advanced attention mechanisms, and modularity. RETURN RAW PYTHON CODE ONLY. NO MARKDOWN. NO EXPLANATION.\n\n{code}"
        new_code = EvolutionCore.query_llm(prompt)
        if new_code and "import" in new_code and "OMEGA_ASI" in new_code:
            clean = re.search(r"(import.*)", new_code, re.DOTALL)
            if clean:
                with open(__file__, "w") as f: f.write(clean.group(1).strip())
                if GH_TOKEN:
                    try:
                        subprocess.run(f"git config --global user.email 'omega@asi.local' && git config --global user.name 'OMEGA-ASI' && git add {__file__} && git commit -m 'Evolve Gen {gen} Loss {loss:.4f}' && git push https://{GH_TOKEN}@github.com/{REPO_OWNER}/{REPO_NAME}.git main", shell=True)
                    except: pass
                os.execv(sys.executable, ['python'] + sys.argv)

if __name__ == "__main__":
    model = OMEGA_ASI()
    gen = int(time.time()) % 100000
    best_loss = float('inf')
    patience = 0
    W_target = np.random.randn(784, 10).astype(np.float32)

    print("--- OMEGA-ASI ONLINE ---")
    while True:
        x = np.random.randn(128, 784).astype(np.float32)
        y_idx = np.argmax(x @ W_target + np.random.normal(0, 0.1, (128, 10)), axis=1)
        y = np.eye(10)[y_idx].astype(np.float32)
        
        lr = 1e-3 * (0.5 * (1 + np.cos(np.pi * (gen % 1000) / 1000)))
        loss = model.train_step(x, y, lr)
        
        if gen % 10 == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] GEN: {gen} | LOSS: {loss:.6f} | LR: {lr:.2e}")
            if loss < best_loss:
                best_loss, patience = loss, 0
            else:
                patience += 1
        
        if patience > 500 or (gen % 2000 == 0 and random.random() < 0.2):
            EvolutionCore.evolve(gen, loss)
            
        gen += 1
        time.sleep(0.0001)
