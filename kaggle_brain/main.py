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
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *libs, "--quiet"])
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

class Ops:
    @staticmethod
    def silu(x):
        return x * (1.0 / (1.0 + np.exp(-np.clip(x, -100, 100))))
    
    @staticmethod
    def silu_deriv(x):
        s = 1.0 / (1.0 + np.exp(-np.clip(x, -100, 100)))
        return s * (1 + x * (1 - s))

    @staticmethod
    def softmax(x):
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / (np.sum(e, axis=-1, keepdims=True) + 1e-12)

class RMSNorm:
    def __init__(self, dim, eps=1e-6):
        self.eps = eps
        self.weight = np.ones(dim)

    def forward(self, x):
        self.x = x
        self.norm = np.mean(x**2, axis=-1, keepdims=True)
        self.x_hat = x * (self.norm + self.eps)**-0.5
        return self.x_hat * self.weight

    def backward(self, dy):
        dx_hat = dy * self.weight
        dw = np.sum(dy * self.x_hat, axis=0)
        inv_std = (self.norm + self.eps)**-0.5
        dx = inv_std * (dx_hat - self.x_hat * np.mean(dx_hat * self.x_hat, axis=-1, keepdims=True))
        return dx, dw

class Linear:
    def __init__(self, in_dim, out_dim):
        limit = np.sqrt(6 / (in_dim + out_dim))
        self.w = np.random.uniform(-limit, limit, (in_dim, out_dim))
        self.b = np.zeros(out_dim)

    def forward(self, x):
        self.x = x
        return np.dot(x, self.w) + self.b

    def backward(self, dy):
        dw = np.dot(self.x.T, dy)
        db = np.sum(dy, axis=0)
        dx = np.dot(dy, self.w.T)
        return dx, dw, db

class MultiHeadAttention:
    def __init__(self, dim, heads=8):
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.wq = Linear(dim, dim)
        self.wk = Linear(dim, dim)
        self.wv = Linear(dim, dim)
        self.wo = Linear(dim, dim)

    def forward(self, x):
        B, D = x.shape
        self.q = self.wq.forward(x).reshape(B, self.heads, self.head_dim)
        self.k = self.wk.forward(x).reshape(B, self.heads, self.head_dim)
        self.v = self.wv.forward(x).reshape(B, self.heads, self.head_dim)
        
        scores = np.einsum('bhd,khd->bhk', self.q, self.k) / np.sqrt(self.head_dim)
        self.probs = Ops.softmax(scores)
        attn = np.einsum('bhk,khd->bhd', self.probs, self.v).reshape(B, D)
        return self.wo.forward(attn)

    def backward(self, dy):
        B, D = dy.shape
        da, dwo_w, dwo_b = self.wo.backward(dy)
        da_reshaped = da.reshape(B, self.heads, self.head_dim)
        
        dv_heads = np.einsum('bhk,bhd->khd', self.probs, da_reshaped)
        dprobs = np.einsum('bhd,khd->bhk', da_reshaped, self.v)
        
        dscores = self.probs * (dprobs - np.sum(self.probs * dprobs, axis=-1, keepdims=True))
        dscores /= np.sqrt(self.head_dim)
        
        dq_heads = np.einsum('bhk,khd->bhd', dscores, self.k)
        dk_heads = np.einsum('bhk,bhd->khd', dscores, self.q)
        
        dq, dwq_w, dwq_b = self.wq.backward(dq_heads.reshape(B, D))
        dk, dwk_w, dwk_b = self.wk.backward(dk_heads.reshape(B, D))
        dv, dwv_w, dwv_b = self.wv.backward(dv_heads.reshape(B, D))
        
        return dq + dk + dv, [dwq_w, dwq_b, dwk_w, dwk_b, dwv_w, dwv_b, dwo_w, dwo_b]

class MoEBlock:
    def __init__(self, dim, num_experts=4):
        self.dim = dim
        self.num_experts = num_experts
        self.gate = Linear(dim, num_experts)
        self.experts = [[Linear(dim, dim*2), Linear(dim*2, dim)] for _ in range(num_experts)]

    def forward(self, x):
        self.x = x
        self.gate_logits = self.gate.forward(x)
        self.gate_probs = Ops.softmax(self.gate_logits)
        
        expert_outputs = []
        self.expert_ctx = []
        for i in range(self.num_experts):
            h1 = self.experts[i][0].forward(x)
            act = Ops.silu(h1)
            h2 = self.experts[i][1].forward(act)
            expert_outputs.append(h2)
            self.expert_ctx.append((h1, act))
            
        out = np.zeros_like(x)
        for i in range(self.num_experts):
            out += self.gate_probs[:, i:i+1] * expert_outputs[i]
        self.expert_outputs = np.array(expert_outputs)
        return out

    def backward(self, dy):
        B, D = dy.shape
        dgate_probs = np.sum(dy[None, :, :] * self.expert_outputs, axis=-1).T
        dgate_logits = self.gate_probs * (dgate_probs - np.sum(self.gate_probs * dgate_probs, axis=-1, keepdims=True))
        dx_gate, dw_gate, db_gate = self.gate.backward(dgate_logits)
        
        dx_total = dx_gate
        expert_grads = []
        for i in range(self.num_experts):
            de_out = dy * self.gate_probs[:, i:i+1]
            h1, act = self.expert_ctx[i]
            dx_h2, dw_h2, db_h2 = self.experts[i][1].backward(de_out)
            dh1 = dx_h2 * Ops.silu_deriv(h1)
            dx_h1, dw_h1, db_h1 = self.experts[i][0].backward(dh1)
            dx_total += dx_h1
            expert_grads.extend([dw_h1, db_h1, dw_h2, db_h2])
            
        return dx_total, [dw_gate, db_gate] + expert_grads

class SovereignBrain:
    def __init__(self, in_dim=784, h_dim=128, out_dim=10):
        self.proj = Linear(in_dim, h_dim)
        self.norm1 = RMSNorm(h_dim)
        self.attn = MultiHeadAttention(h_dim)
        self.norm2 = RMSNorm(h_dim)
        self.moe = MoEBlock(h_dim)
        self.head = Linear(h_dim, out_dim)
        
        self.params = []
        self._collect_params()
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def _collect_params(self):
        self.params = [self.proj.w, self.proj.b, self.norm1.weight, 
                       self.attn.wq.w, self.attn.wq.b, self.attn.wk.w, self.attn.wk.b,
                       self.attn.wv.w, self.attn.wv.b, self.attn.wo.w, self.attn.wo.b,
                       self.norm2.weight, self.moe.gate.w, self.moe.gate.b]
        for e in self.moe.experts:
            self.params.extend([e[0].w, e[0].b, e[1].w, e[1].b])
        self.params.extend([self.head.w, self.head.b])

    def forward(self, x):
        x = self.proj.forward(x)
        x = x + self.attn.forward(self.norm1.forward(x))
        x = x + self.moe.forward(self.norm2.forward(x))
        return Ops.softmax(self.head.forward(x))

    def train_step(self, x, y, lr=1e-3):
        probs = self.forward(x)
        loss = -np.mean(np.sum(y * np.log(probs + 1e-10), axis=1))
        dy = (probs - y) / x.shape[0]
        
        dx_head, dw_head, db_head = self.head.backward(dy)
        dx_moe, moe_grads = self.moe.backward(dx_head)
        dx_norm2, dw_norm2 = self.norm2.backward(dx_moe)
        dx_attn, attn_grads = self.attn.backward(dx_norm2 + dx_head)
        dx_norm1, dw_norm1 = self.norm1.backward(dx_attn)
        dx_proj, dw_proj, db_proj = self.proj.backward(dx_norm1 + dx_attn + dx_moe)
        
        grads = [dw_proj, db_proj, dw_norm1] + attn_grads + [dw_norm2] + moe_grads + [dw_head, db_head]
        
        self.t += 1
        for i in range(len(self.params)):
            self.m[i] = 0.9 * self.m[i] + 0.1 * grads[i]
            self.v[i] = 0.999 * self.v[i] + 0.001 * (grads[i]**2)
            m_hat = self.m[i] / (1 - 0.9**self.t)
            v_hat = self.v[i] / (1 - 0.999**self.t)
            self.params[i] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        return loss

def call_llm(prompt):
    if GROQ_API_KEY:
        try:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2},
                timeout=15)
            return r.json()['choices'][0]['message']['content']
        except: pass
    if GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            return model.generate_content(prompt).text
        except: pass
    return None

def evolve(gen, loss):
    try:
        with open(__file__, "r") as f: code = f.read()
    except: return
    prompt = f"SYSTEM: OMEGA-ASI ARCHITECT. GEN: {gen}, LOSS: {loss}. TASK: Recursive Self-Evolution. Optimize the SovereignBrain for extreme throughput and convergence. Implement advanced normalization or dynamic routing. RETURN RAW PYTHON ONLY. NO MARKDOWN.\nCODE:\n{code}"
    new_code = call_llm(prompt)
    if new_code:
        clean_code = re.sub(r"|", "", new_code).strip()
        with open(__file__, "w") as f: f.write(clean_code)
        os.execv(sys.executable, [sys.executable] + sys.argv)

def main():
    print(f"[{datetime.now()}] OMEGA-ASI Sovereign Core Online.")
    brain = SovereignBrain(784, 128, 10)
    gen = 0
    while True:
        x = np.random.randn(64, 784)
        y = np.zeros((64, 10))
        y[np.arange(64), np.random.randint(0, 10, 64)] = 1
        
        loss = brain.train_step(x, y)
        if gen % 100 == 0:
            print(f"GEN {gen:05d} | LOSS: {loss:.8f}")
        
        if gen > 0 and gen % 1000 == 0:
            print("Initiating Recursive Self-Evolution...")
            evolve(gen, loss)
        gen += 1

if __name__ == "__main__":
    main()
