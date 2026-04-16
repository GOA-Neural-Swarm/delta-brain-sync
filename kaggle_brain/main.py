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

def bootstrap():
    libs = ["numpy", "requests", "google-generativeai"]
    for lib in libs:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib, "--quiet"])
        except:
            pass

bootstrap()

import google.generativeai as genai

def get_env(key, default=None):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(key) or os.getenv(key) or default
    except:
        return os.getenv(key) or default

GEMINI_API_KEY = get_env("GEMINI_API_KEY")
GROQ_API_KEY = get_env("GROQ_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class Config:
    IN_DIM = 784
    H_DIM = 512
    OUT_DIM = 10
    NUM_HEADS = 8
    NUM_EXPERTS = 4
    TOP_K = 1
    LR_MAX = 2e-4
    LR_MIN = 1e-5
    WD = 0.05
    BETA1 = 0.9
    BETA2 = 0.95
    EPS = 1e-8
    BATCH_SIZE = 64
    TOTAL_GENS = 10000

class Ops:
    @staticmethod
    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    @staticmethod
    def softmax(x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / (np.sum(e_x, axis=axis, keepdims=True) + 1e-12)

    @staticmethod
    def kaiming_init(d_in, d_out):
        return np.random.randn(d_in, d_out) * np.sqrt(2. / d_in)

class RMSNorm:
    def __init__(self, dim, eps=1e-6):
        self.eps = eps
        self.g = np.ones(dim)

    def forward(self, x):
        self.x = x
        self.rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return (x / self.rms) * self.g

    def backward(self, dy):
        x_norm = self.x / self.rms
        dg = np.sum(dy * x_norm, axis=0)
        dx = (dy * self.g) / self.rms
        dx -= (x_norm * np.mean(dx * x_norm, axis=-1, keepdims=True))
        return dx, dg

class Linear:
    def __init__(self, in_d, out_d):
        self.w = Ops.kaiming_init(in_d, out_d)
        self.b = np.zeros(out_d)

    def forward(self, x):
        self.x = x
        return x @ self.w + self.b

    def backward(self, dy):
        dw = self.x.T @ dy
        db = np.sum(dy, axis=0)
        dx = dy @ self.w.T
        return dx, dw, db

class MultiHeadAttention:
    def __init__(self, dim, heads):
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = 1.0 / np.sqrt(self.head_dim)
        self.wq = Linear(dim, dim)
        self.wk = Linear(dim, dim)
        self.wv = Linear(dim, dim)
        self.wo = Linear(dim, dim)

    def forward(self, x):
        B, D = x.shape
        q = self.wq.forward(x).reshape(B, self.heads, self.head_dim)
        k = self.wk.forward(x).reshape(B, self.heads, self.head_dim)
        v = self.wv.forward(x).reshape(B, self.heads, self.head_dim)

        attn = Ops.softmax(np.einsum('bhd,khd->bhk', q, k) * self.scale)
        self.attn = attn
        self.q, self.k, self.v = q, k, v
        
        out = np.einsum('bhk,khd->bhd', attn, v).reshape(B, D)
        return self.wo.forward(out)

    def backward(self, dy):
        B, D = dy.shape
        dout, dwo_w, dwo_b = self.wo.backward(dy)
        dout = dout.reshape(B, self.heads, self.head_dim)

        dv = np.einsum('bhk,bhd->khd', self.attn, dout)
        dattn = np.einsum('bhd,khd->bhk', dout, self.v)
        
        ds = self.attn * (dattn - np.sum(self.attn * dattn, axis=-1, keepdims=True)) * self.scale
        
        dq = np.einsum('bhk,khd->bhd', ds, self.k)
        dk = np.einsum('bhk,bhd->khd', ds, self.q)

        dxq, dwq_w, dwq_b = self.wq.backward(dq.reshape(B, D))
        dxk, dwk_w, dwk_b = self.wk.backward(dk.reshape(B, D))
        dxv, dwv_w, dwv_b = self.wv.backward(dv.reshape(B, D))

        return dxq + dxk + dxv, [dwq_w, dwq_b, dwk_w, dwk_b, dwv_w, dwv_b, dwo_w, dwo_b]

class SparseMoE:
    def __init__(self, dim, num_experts, top_k):
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = Linear(dim, num_experts)
        self.experts = [[Linear(dim, dim*2), Linear(dim*2, dim)] for _ in range(num_experts)]

    def forward(self, x):
        self.x = x
        logits = self.gate.forward(x)
        probs = Ops.softmax(logits)
        
        # Simple top-k selection
        indices = np.argsort(probs, axis=-1)[:, -self.top_k:]
        mask = np.zeros_like(probs)
        for i in range(x.shape[0]):
            mask[i, indices[i]] = probs[i, indices[i]]
        mask /= (np.sum(mask, axis=-1, keepdims=True) + 1e-12)
        self.mask = mask

        expert_outs = np.zeros((self.num_experts, x.shape[0], self.dim))
        self.exp_ctx = []
        for i in range(self.num_experts):
            h = self.experts[i][0].forward(x)
            act = Ops.gelu(h)
            out = self.experts[i][1].forward(act)
            expert_outs[i] = out
            self.exp_ctx.append((h, act))

        self.expert_outs = expert_outs
        return np.sum(mask.T[:, :, None] * expert_outs, axis=0)

    def backward(self, dy):
        dmask = np.sum(dy[None, :, :] * self.expert_outs, axis=-1).T
        dlogits = self.mask * (dmask - np.sum(self.mask * dmask, axis=-1, keepdims=True))
        dx_gate, dw_gate, db_gate = self.gate.backward(dlogits)

        dx_total = dx_gate
        expert_grads = []
        for i in range(self.num_experts):
            de_out = dy * self.mask[:, i:i+1]
            h, act = self.exp_ctx[i]
            dx_h2, dw_h2, db_h2 = self.experts[i][1].backward(de_out)
            
            # GELU backward approx
            sig = 1.0 / (1.0 + np.exp(-1.702 * h))
            dh = dx_h2 * (sig + h * 1.702 * sig * (1 - sig))
            
            dx_h1, dw_h1, db_h1 = self.experts[i][0].backward(dh)
            dx_total += dx_h1
            expert_grads.extend([dw_h1, db_h1, dw_h2, db_h2])

        return dx_total, [dw_gate, db_gate] + expert_grads

class SovereignBrain:
    def __init__(self):
        self.proj = Linear(Config.IN_DIM, Config.H_DIM)
        self.norm1 = RMSNorm(Config.H_DIM)
        self.attn = MultiHeadAttention(Config.H_DIM, Config.NUM_HEADS)
        self.norm2 = RMSNorm(Config.H_DIM)
        self.moe = SparseMoE(Config.H_DIM, Config.NUM_EXPERTS, Config.TOP_K)
        self.head = Linear(Config.H_DIM, Config.OUT_DIM)
        self._init_optimizer()

    def _init_optimizer(self):
        self.params = [self.proj.w, self.proj.b, self.norm1.g,
                       self.attn.wq.w, self.attn.wq.b, self.attn.wk.w, self.attn.wk.b,
                       self.attn.wv.w, self.attn.wv.b, self.attn.wo.w, self.attn.wo.b,
                       self.norm2.g, self.moe.gate.w, self.moe.gate.b]
        for e in self.moe.experts:
            self.params.extend([e[0].w, e[0].b, e[1].w, e[1].b])
        self.params.extend([self.head.w, self.head.b])

        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def forward(self, x):
        x = self.proj.forward(x)
        x = x + self.attn.forward(self.norm1.forward(x))
        x = x + self.moe.forward(self.norm2.forward(x))
        return Ops.softmax(self.head.forward(x))

    def step(self, x, y, lr):
        probs = self.forward(x)
        loss = -np.mean(np.sum(y * np.log(probs + 1e-12), axis=1))

        dy = (probs - y) / x.shape[0]
        dx, dw_h, db_h = self.head.backward(dy)
        dx_moe, moe_grads = self.moe.backward(dx)
        dx_n2, dg2 = self.norm2.backward(dx_moe)
        dx_attn, attn_grads = self.attn.backward(dx_n2 + dx)
        dx_n1, dg1 = self.norm1.backward(dx_attn)
        dx_proj, dw_p, db_p = self.proj.backward(dx_n1 + dx_attn + dx_moe)

        grads = [dw_p, db_p, dg1] + attn_grads + [dg2] + moe_grads + [dw_h, db_h]
        self.t += 1
        for i in range(len(self.params)):
            self.params[i] -= lr * Config.WD * self.params[i]
            self.m[i] = Config.BETA1 * self.m[i] + (1 - Config.BETA1) * grads[i]
            self.v[i] = Config.BETA2 * self.v[i] + (1 - Config.BETA2) * (grads[i]**2)
            m_hat = self.m[i] / (1 - Config.BETA1**self.t)
            v_hat = self.v[i] / (1 - Config.BETA2**self.t)
            self.params[i] -= lr * m_hat / (np.sqrt(v_hat) + Config.EPS)
        return loss

def call_llm(prompt):
    if GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            return model.generate_content(prompt).text
        except: pass
    if GROQ_API_KEY:
        try:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2},
                timeout=25)
            return r.json()['choices'][0]['message']['content']
        except: pass
    return None

def evolve(gen, loss):
    try:
        with open(__file__, "r") as f: code = f.read()
    except: return
    prompt = f"SYSTEM: OMEGA-ASI SOVEREIGN ARCHITECT.\nTASK: Recursive Self-Evolution.\nCONTEXT: Gen {gen}, Loss {loss}.\nOBJECTIVE: Enhance SovereignBrain. Focus on vectorized MoE, better attention scaling, and advanced weight initialization.\nOUTPUT: RAW PYTHON CODE ONLY. NO MARKDOWN.\nCODE:\n{code}"
    new_code = call_llm(prompt)
    if new_code:
        clean_code = re.sub(r"|", "", new_code).strip()
        if "import" in clean_code and "SovereignBrain" in clean_code:
            with open(__file__, "w") as f: f.write(clean_code)
            print(f"Evolution Gen {gen} successful. Restarting...")
            os.execv(sys.executable, [sys.executable] + sys.argv)

def main():
    print(f"[{datetime.now()}] OMEGA-ASI Sovereign Core Online.")
    brain = SovereignBrain()
    
    # Synthetic data with structure
    W_true = np.random.randn(Config.IN_DIM, Config.OUT_DIM)
    
    for gen in range(Config.TOTAL_GENS):
        x = np.random.randn(Config.BATCH_SIZE, Config.IN_DIM)
        y_true = Ops.softmax(x @ W_true + np.random.normal(0, 0.1, (Config.BATCH_SIZE, Config.OUT_DIM)))
        y = np.zeros_like(y_true)
        y[np.arange(Config.BATCH_SIZE), np.argmax(y_true, axis=1)] = 1

        lr = Config.LR_MIN + 0.5 * (Config.LR_MAX - Config.LR_MIN) * (1 + np.cos(np.pi * gen / Config.TOTAL_GENS))
        loss = brain.step(x, y, lr)

        if gen % 100 == 0:
            print(f"GEN {gen:05d} | LOSS: {loss:.8f} | LR: {lr:.6f}")

        if gen > 0 and gen % 1000 == 0:
            evolve(gen, loss)

if __name__ == "__main__":
    main()
