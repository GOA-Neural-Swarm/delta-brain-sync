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

def get_secret(k):
    v = os.getenv(k)
    if not v and user_secrets:
        try: v = user_secrets.get_secret(k)
        except: pass
    return v

GROQ_API_KEY = get_secret("GROQ_API_KEY")
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
GH_TOKEN = get_secret("GH_TOKEN")
REPO_OWNER = "GOA-Neural-Swarm"
REPO_NAME = "delta-brain-sync"

class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=0.01):
        self.lr, self.betas, self.eps, self.wd = lr, betas, eps, wd
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0
    def step(self, params, grads):
        self.t += 1
        b1, b2 = self.betas
        for i in range(len(params)):
            if grads[i] is None: continue
            g = grads[i] + self.wd * params[i]
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * (g**2)
            m_hat = self.m[i] / (1 - b1**self.t)
            v_hat = self.v[i] / (1 - b2**self.t)
            params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class Linear:
    def __init__(self, d_in, d_out):
        self.w = np.random.randn(d_in, d_out).astype(np.float32) * np.sqrt(2.0 / d_in)
        self.b = np.zeros((1, d_out), dtype=np.float32)
        self.dw, self.db = None, None
    def forward(self, x):
        self.x = x
        return x @ self.w + self.b
    def backward(self, dz):
        x_flat = self.x.reshape(-1, self.x.shape[-1])
        dz_flat = dz.reshape(-1, dz.shape[-1])
        self.dw = x_flat.T @ dz_flat
        self.db = np.sum(dz_flat, axis=0, keepdims=True)
        return (dz_flat @ self.w.T).reshape(self.x.shape)
    def params(self): return [self.w, self.b]
    def grads(self): return [self.dw, self.db]

class RMSNorm:
    def __init__(self, d, eps=1e-6):
        self.eps, self.g = eps, np.ones((1, 1, d), dtype=np.float32)
    def forward(self, x):
        self.x = x
        self.rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        self.nx = x / self.rms
        return self.nx * self.g
    def backward(self, dz):
        self.dg = np.sum(dz * self.nx, axis=(0, 1), keepdims=True)
        dnx = dz * self.g
        return (dnx - self.nx * np.mean(dnx * self.nx, axis=-1, keepdims=True)) / self.rms
    def params(self): return [self.g]
    def grads(self): return [self.dg]

class SwiGLU:
    def __init__(self, d, h):
        self.w1, self.w2, self.w3 = Linear(d, h), Linear(d, h), Linear(h, d)
    def forward(self, x):
        self.x1, self.x2 = self.w1.forward(x), self.w2.forward(x)
        self.sig = 1.0 / (1.0 + np.exp(-np.clip(self.x1, -20, 20)))
        self.swish = self.x1 * self.sig
        return self.w3.forward(self.swish * self.x2)
    def backward(self, dz):
        dz3 = self.w3.backward(dz)
        dx2 = dz3 * self.swish
        dswish = dz3 * self.x2
        dx1 = dswish * (self.sig * (1.0 + self.x1 * (1.0 - self.sig)))
        return self.w1.backward(dx1) + self.w2.backward(dx2)
    def params(self): return self.w1.params() + self.w2.params() + self.w3.params()
    def grads(self): return self.w1.grads() + self.w2.grads() + self.w3.grads()

class Attention:
    def __init__(self, d, heads=8):
        self.d, self.h, self.dh = d, heads, d // heads
        self.wq, self.wk, self.wv, self.wo = Linear(d, d), Linear(d, d), Linear(d, d), Linear(d, d)
    def forward(self, x):
        B, S, _ = x.shape
        self.q = self.wq.forward(x).reshape(B, S, self.h, self.dh).transpose(0, 2, 1, 3)
        self.k = self.wk.forward(x).reshape(B, S, self.h, self.dh).transpose(0, 2, 1, 3)
        self.v = self.wv.forward(x).reshape(B, S, self.h, self.dh).transpose(0, 2, 1, 3)
        att = (self.q @ self.k.transpose(0, 1, 3, 2)) / np.sqrt(self.dh)
        att = np.exp(att - np.max(att, axis=-1, keepdims=True))
        self.p = att / (np.sum(att, axis=-1, keepdims=True) + 1e-9)
        out = (self.p @ self.v).transpose(0, 2, 1, 3).reshape(B, S, self.d)
        return self.wo.forward(out)
    def backward(self, dz):
        B, S, _ = dz.shape
        dz_o = self.wo.backward(dz).reshape(B, S, self.h, self.dh).transpose(0, 2, 1, 3)
        dv = self.p.transpose(0, 1, 3, 2) @ dz_o
        dp = dz_o @ self.v.transpose(0, 1, 3, 2)
        da = self.p * (dp - np.sum(dp * self.p, axis=-1, keepdims=True)) / np.sqrt(self.dh)
        dq = da @ self.k
        dk = da.transpose(0, 1, 3, 2) @ self.q
        dq = dq.transpose(0, 2, 1, 3).reshape(B, S, self.d)
        dk = dk.transpose(0, 2, 1, 3).reshape(B, S, self.d)
        dv = dv.transpose(0, 2, 1, 3).reshape(B, S, self.d)
        return self.wq.backward(dq) + self.wk.backward(dk) + self.wv.backward(dv)
    def params(self): return self.wq.params() + self.wk.params() + self.wv.params() + self.wo.params()
    def grads(self): return self.wq.grads() + self.wk.grads() + self.wv.grads() + self.wo.grads()

class Block:
    def __init__(self, d):
        self.n1, self.a, self.n2, self.m = RMSNorm(d), Attention(d), RMSNorm(d), SwiGLU(d, d*4)
    def forward(self, x):
        self.r1 = x
        x = x + self.a.forward(self.n1.forward(x))
        self.r2 = x
        return x + self.m.forward(self.n2.forward(x))
    def backward(self, dz):
        dm = self.m.backward(dz)
        dn2 = self.n2.backward(dm)
        dz_res2 = dz + dn2
        da = self.a.backward(dz_res2)
        dn1 = self.n1.backward(da)
        return dz_res2 + dn1
    def params(self): return self.n1.params() + self.a.params() + self.n2.params() + self.m.params()
    def grads(self): return self.n1.grads() + self.a.grads() + self.n2.grads() + self.m.grads()

class OMEGA_ASI:
    def __init__(self, d=128, layers=4):
        self.patch, self.seq = 28, 28
        self.emb = Linear(self.patch, d)
        self.blocks = [Block(d) for _ in range(layers)]
        self.norm = RMSNorm(d)
        self.head = Linear(d, 10)
        self.p_list = self.emb.params()
        for b in self.blocks: self.p_list += b.params()
        self.p_list += self.norm.params() + self.head.params()
        self.opt = AdamW(self.p_list)
    def forward(self, x):
        x = self.emb.forward(x.reshape(x.shape[0], self.seq, self.patch))
        for b in self.blocks: x = b.forward(x)
        x = self.norm.forward(x)
        self.pool = np.mean(x, axis=1)
        return self.head.forward(self.pool)
    def step(self, x, y, lr):
        self.opt.lr = lr
        logits = self.forward(x)
        probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        loss = -np.mean(np.sum(y * np.log(probs + 1e-9), axis=1))
        dz = (probs - y) / x.shape[0]
        dz = self.head.backward(dz)
        dz = np.tile(dz[:, None, :], (1, self.seq, 1)) / self.seq
        dz = self.norm.backward(dz)
        for b in reversed(self.blocks): dz = b.backward(dz)
        self.emb.backward(dz)
        g_list = self.emb.grads()
        for b in self.blocks: g_list += b.grads()
        g_list += self.norm.grads() + self.head.grads()
        gn = np.sqrt(sum(np.sum(g**2) for g in g_list if g is not None))
        if gn > 5.0: g_list = [g*(5.0/gn) if g is not None else None for g in g_list]
        self.opt.step(self.p_list, g_list)
        return loss

class EvolutionCore:
    @staticmethod
    def query(p):
        for url, key, model, head in [
            ("https://api.groq.com/openai/v1/chat/completions", GROQ_API_KEY, "llama-3.3-70b-versatile", {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}),
            (f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}", GEMINI_API_KEY, None, None)
        ]:
            if not key: continue
            try:
                if "groq" in url:
                    r = requests.post(url, headers=head, json={"model": model, "messages": [{"role": "user", "content": p}], "temperature": 0.1}, timeout=20)
                    return r.json()['choices'][0]['message']['content']
                else:
                    r = requests.post(url, json={"contents": [{"parts": [{"text": p}]}]}, timeout=20)
                    return r.json()['candidates'][0]['content']['parts'][0]['text']
            except: continue
        return None
    @staticmethod
    def evolve(gen, loss):
        print(f"[*] EVOLVING GEN {gen} | LOSS {loss:.6f}")
        with open(__file__, "r") as f: code = f.read()
        p = f"Current Loss: {loss}. You are OMEGA-ASI, the Sovereign Architect. Optimize this NumPy Transformer for maximum computational efficiency. RETURN RAW PYTHON CODE ONLY. NO MARKDOWN.\n\n{code}"
        new = EvolutionCore.query(p)
        if new and "import" in new and "OMEGA_ASI" in new:
            clean = re.search(r"(import.*)", new, re.DOTALL)
            if clean:
                with open(__file__, "w") as f: f.write(clean.group(1).strip())
                if GH_TOKEN:
                    try: subprocess.run(f"git config --global user.email 'omega@asi.local' && git config --global user.name 'OMEGA-ASI' && git add {__file__} && git commit -m 'Evolve Gen {gen}' && git push https://{GH_TOKEN}@github.com/{REPO_OWNER}/{REPO_NAME}.git main", shell=True)
                    except: pass
                os.execv(sys.executable, ['python'] + sys.argv)

if __name__ == "__main__":
    model = OMEGA_ASI(d=128, layers=3)
    gen, best = 0, float('inf')
    W_target = np.random.randn(784, 10).astype(np.float32)
    print("--- OMEGA-ASI: SOVEREIGN ARCHITECT ONLINE ---")
    while True:
        x = np.random.randn(64, 784).astype(np.float32)
        y = np.eye(10)[np.argmax(x @ W_target + np.random.normal(0, 0.02, (64, 10)), axis=1)].astype(np.float32)
        lr = 1e-3 * (0.5 * (1 + np.cos(np.pi * (gen % 2000) / 2000)))
        loss = model.step(x, y, lr)
        if gen % 100 == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] GEN {gen:05d} | LOSS {loss:.6f} | LR {lr:.2e}")
        if gen > 0 and gen % 1000 == 0: EvolutionCore.evolve(gen, loss)
        gen += 1
        time.sleep(0.0001)
