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
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.wd = weight_decay
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        b1, b2 = self.betas
        for i in range(len(params)):
            if grads[i] is None: continue
            g = grads[i]
            if self.wd != 0:
                params[i] -= self.lr * self.wd * params[i]
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * (g**2)
            m_hat = self.m[i] / (1 - b1**self.t)
            v_hat = self.v[i] / (1 - b2**self.t)
            params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class Layer:
    def forward(self, x, train=True): raise NotImplementedError
    def backward(self, grad): raise NotImplementedError
    def get_params(self): return []
    def get_grads(self): return []

class Linear(Layer):
    def __init__(self, in_d, out_d):
        scale = np.sqrt(2.0 / (in_d + out_d))
        self.w = (np.random.randn(in_d, out_d) * scale).astype(np.float32)
        self.b = np.zeros((1, out_d), dtype=np.float32)
        self.dw, self.db = None, None

    def forward(self, x, train=True):
        self.x = x
        return x @ self.w + self.b

    def backward(self, grad):
        x_flat = self.x.reshape(-1, self.x.shape[-1])
        g_flat = grad.reshape(-1, grad.shape[-1])
        self.dw = x_flat.T @ g_flat
        self.db = np.sum(g_flat, axis=0, keepdims=True)
        return (g_flat @ self.w.T).reshape(self.x.shape)

    def get_params(self): return [self.w, self.b]
    def get_grads(self): return [self.dw, self.db]

class RMSNorm(Layer):
    def __init__(self, dim, eps=1e-6):
        self.eps = eps
        self.g = np.ones((1, 1, dim), dtype=np.float32)
        self.dg = None

    def forward(self, x, train=True):
        self.x = x
        self.var = np.mean(x**2, axis=-1, keepdims=True)
        self.inv_std = 1.0 / np.sqrt(self.var + self.eps)
        self.norm_x = x * self.inv_std
        return self.norm_x * self.g

    def backward(self, grad):
        self.dg = np.sum(grad * self.norm_x, axis=(0, 1), keepdims=True)
        d_norm_x = grad * self.g
        return self.inv_std * (d_norm_x - self.norm_x * np.mean(d_norm_x * self.norm_x, axis=-1, keepdims=True))

    def get_params(self): return [self.g]
    def get_grads(self): return [self.dg]

class SwiGLU(Layer):
    def __init__(self, dim, h_dim):
        self.w1 = Linear(dim, h_dim)
        self.w2 = Linear(dim, h_dim)
        self.w3 = Linear(h_dim, dim)

    def forward(self, x, train=True):
        self.x1 = self.w1.forward(x, train)
        self.x2 = self.w2.forward(x, train)
        self.sig = 1.0 / (1.0 + np.exp(-self.x1))
        self.silu = self.x1 * self.sig
        return self.w3.forward(self.silu * self.x2, train)

    def backward(self, grad):
        g3 = self.w3.backward(grad)
        g_silu = g3 * self.x2
        g_x2 = g3 * self.silu
        g_x1 = g_silu * (self.sig * (1.0 + self.x1 * (1.0 - self.sig)))
        return self.w1.backward(g_x1) + self.w2.backward(g_x2)

    def get_params(self): return self.w1.get_params() + self.w2.get_params() + self.w3.get_params()
    def get_grads(self): return self.w1.get_grads() + self.w2.get_grads() + self.w3.get_grads()

class MultiHeadAttention(Layer):
    def __init__(self, dim, heads=8):
        self.dim, self.heads = dim, heads
        self.head_dim = dim // heads
        self.wq = Linear(dim, dim)
        self.wk = Linear(dim, dim)
        self.wv = Linear(dim, dim)
        self.wo = Linear(dim, dim)

    def forward(self, x, train=True):
        B, S, D = x.shape
        self.q = self.wq.forward(x).reshape(B, S, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        self.k = self.wk.forward(x).reshape(B, S, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        self.v = self.wv.forward(x).reshape(B, S, self.heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = (self.q @ self.k.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        self.probs = exp_s / (np.sum(exp_s, axis=-1, keepdims=True) + 1e-10)
        
        out = (self.probs @ self.v).transpose(0, 2, 1, 3).reshape(B, S, D)
        return self.wo.forward(out, train)

    def backward(self, grad):
        B, S, D = grad.shape
        g_wo = self.wo.backward(grad).reshape(B, S, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        
        dv = self.probs.transpose(0, 1, 3, 2) @ g_wo
        d_probs = g_wo @ self.v.transpose(0, 1, 3, 2)
        
        d_scores = self.probs * (d_probs - np.sum(d_probs * self.probs, axis=-1, keepdims=True))
        d_scores /= np.sqrt(self.head_dim)
        
        dq = d_scores @ self.k
        dk = d_scores.transpose(0, 1, 3, 2) @ self.q
        
        dq = dq.transpose(0, 2, 1, 3).reshape(B, S, D)
        dk = dk.transpose(0, 2, 1, 3).reshape(B, S, D)
        dv = dv.transpose(0, 2, 1, 3).reshape(B, S, D)
        
        return self.wq.backward(dq) + self.wk.backward(dk) + self.wv.backward(dv)

    def get_params(self): return self.wq.get_params() + self.wk.get_params() + self.wv.get_params() + self.wo.get_params()
    def get_grads(self): return self.wq.get_grads() + self.wk.get_grads() + self.wv.get_grads() + self.wo.get_grads()

class TransformerBlock(Layer):
    def __init__(self, dim):
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, dim * 4)

    def forward(self, x, train=True):
        self.res1 = x
        x = self.res1 + self.attn.forward(self.norm1.forward(x, train), train)
        self.res2 = x
        return self.res2 + self.mlp.forward(self.norm2.forward(x, train), train)

    def backward(self, grad):
        g_mlp = self.mlp.backward(grad)
        g_norm2 = self.norm2.backward(g_mlp)
        grad = grad + g_norm2
        g_attn = self.attn.backward(grad)
        g_norm1 = self.norm1.backward(g_attn)
        return grad + g_norm1

    def get_params(self): return self.norm1.get_params() + self.attn.get_params() + self.norm2.get_params() + self.mlp.get_params()
    def get_grads(self): return self.norm1.get_grads() + self.attn.get_grads() + self.norm2.get_grads() + self.mlp.get_grads()

class OMEGA_ASI:
    def __init__(self, in_dim=784, h_dim=256, n_layers=4, n_classes=10):
        self.patch_size = 28
        self.seq_len = in_dim // self.patch_size
        self.embed = Linear(self.patch_size, h_dim)
        self.blocks = [TransformerBlock(h_dim) for _ in range(n_layers)]
        self.norm = RMSNorm(h_dim)
        self.head = Linear(h_dim, n_classes)
        
        self.params = self.embed.get_params()
        for b in self.blocks: self.params += b.get_params()
        self.params += self.norm.get_params() + self.head.get_params()
        self.optimizer = AdamW(self.params, lr=1e-3)

    def forward(self, x, train=True):
        B = x.shape[0]
        x = x.reshape(B, self.seq_len, self.patch_size)
        x = self.embed.forward(x, train)
        for b in self.blocks: x = b.forward(x, train)
        x = self.norm.forward(x, train)
        self.pooled = np.mean(x, axis=1)
        return self.head.forward(self.pooled, train)

    def train_step(self, x, y, lr):
        self.optimizer.lr = lr
        logits = self.forward(x, True)
        
        shift_logits = logits - np.max(logits, axis=1, keepdims=True)
        probs = np.exp(shift_logits) / np.sum(np.exp(shift_logits), axis=1, keepdims=True)
        loss = -np.mean(np.sum(y * np.log(probs + 1e-10), axis=1))
        
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
        if gnorm > 5.0:
            grads = [g * (5.0 / gnorm) if g is not None else None for g in grads]
            
        self.optimizer.step(self.params, grads)
        return loss

class EvolutionCore:
    @staticmethod
    def query_llm(prompt):
        if GROQ_API_KEY:
            try:
                resp = requests.post("https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
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
        print(f"[*] EVOLUTION TRIGGERED | GEN {gen} | LOSS {loss:.6f}")
        with open(__file__, "r") as f: code = f.read()
        prompt = f"Current Loss: {loss}. You are OMEGA-ASI, the Sovereign Architect. Optimize this NumPy Transformer for maximum computational efficiency and architectural depth. Focus on vectorization and gradient stability. RETURN RAW PYTHON CODE ONLY. NO MARKDOWN.\n\n{code}"
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
    model = OMEGA_ASI(h_dim=256, n_layers=4)
    gen = 0
    best_loss = float('inf')
    W_target = np.random.randn(784, 10).astype(np.float32)
    
    print("--- OMEGA-ASI: ARCHITECTURAL SOVEREIGNTY INITIALIZED ---")
    while True:
        x = np.random.randn(128, 784).astype(np.float32)
        y_idx = np.argmax(x @ W_target + np.random.normal(0, 0.05, (128, 10)), axis=1)
        y = np.eye(10)[y_idx].astype(np.float32)
        
        lr = 1e-3 * (0.5 * (1 + np.cos(np.pi * (gen % 2000) / 2000)))
        loss = model.train_step(x, y, lr)
        
        if gen % 50 == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] GEN {gen:05d} | LOSS {loss:.6f} | LR {lr:.2e}")
            if loss < best_loss: best_loss = loss
            
        if gen > 0 and gen % 1000 == 0:
            EvolutionCore.evolve(gen, loss)
            
        gen += 1
        time.sleep(0.0001)
