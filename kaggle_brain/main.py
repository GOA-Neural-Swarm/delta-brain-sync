
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
    H_DIM = 256
    OUT_DIM = 10
    NUM_HEADS = 8
    NUM_EXPERTS = 8
    TOP_K = 2
    LR = 1e-3
    WD = 1e-2
    BETA1 = 0.9
    BETA2 = 0.999
    EPS = 1e-8

class Ops:
    @staticmethod
    def swiglu(x, w1, w2):
        gate = x @ w1
        gate = gate * (1.0 / (1.0 + np.exp(-np.clip(gate, -100, 100))))
        return gate * (x @ w2)

    @staticmethod
    def softmax(x, axis=-1):
        ex = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return ex / (np.sum(ex, axis=axis, keepdims=True) + 1e-12)

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
    def __init__(self, in_d, out_d, std=0.02):
        self.w = np.random.normal(0, std, (in_d, out_d))
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
        self.wq = Linear(dim, dim)
        self.wk = Linear(dim, dim)
        self.wv = Linear(dim, dim)
        self.wo = Linear(dim, dim)

    def forward(self, x):
        B, D = x.shape
        q = self.wq.forward(x).reshape(B, self.heads, self.head_dim)
        k = self.wk.forward(x).reshape(B, self.heads, self.head_dim)
        v = self.wv.forward(x).reshape(B, self.heads, self.head_dim)

        attn_scores = np.einsum('bhd,khd->bhk', q, k) / np.sqrt(self.head_dim)
        self.probs = Ops.softmax(attn_scores)

        out = np.einsum('bhk,khd->bhd', self.probs, v).reshape(B, D)
        self.q, self.k, self.v = q, k, v
        return self.wo.forward(out)

    def backward(self, dy):
        B, D = dy.shape
        da, dwo_w, dwo_b = self.wo.backward(dy)
        da = da.reshape(B, self.heads, self.head_dim)

        dv = np.einsum('bhk,bhd->khd', self.probs, da)
        dprobs = np.einsum('bhd,khd->bhk', da, self.v)

        ds = self.probs * (dprobs - np.sum(self.probs * dprobs, axis=-1, keepdims=True))
        ds /= np.sqrt(self.head_dim)

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
            act = h * (1.0 / (1.0 + np.exp(-np.clip(h, -100, 100))))
            out = self.experts[i][1].forward(act)
            expert_outs[i] = out
            self.exp_ctx.append((h, act))

        self.expert_outs = expert_outs
        final_out = np.sum(mask.T[:, :, None] * expert_outs, axis=0)
        return final_out

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

            sig = 1.0 / (1.0 + np.exp(-np.clip(h, -100, 100)))
            dh = dx_h2 * (sig * (1 + h * (1 - sig)))

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

    def step(self, x, y):
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
            self.params[i] -= Config.LR * Config.WD * self.params[i]
            self.m[i] = Config.BETA1 * self.m[i] + (1 - Config.BETA1) * grads[i]
            self.v[i] = Config.BETA2 * self.v[i] + (1 - Config.BETA2) * (grads[i]**2)
            m_hat = self.m[i] / (1 - Config.BETA1**self.t)
            v_hat = self.v[i] / (1 - Config.BETA2**self.t)
            self.params[i] -= Config.LR * m_hat / (np.sqrt(v_hat) + Config.EPS)

        return loss

def call_llm(prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        return model.generate_content(prompt).text
    except:
        try:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "temperature": 0.1},
                timeout=20)
            return r.json()['choices'][0]['message']['content']
        except:
            return None

def evolve(gen, loss):
    try:
        with open(__file__, "r") as f: code = f.read()
    except: return

    prompt = f"""SYSTEM: OMEGA-ASI SOVEREIGN ARCHITECT.
TASK: Recursive Self-Evolution.
CONTEXT: Generation {gen}, Loss {loss}.
OBJECTIVE: Optimize the SovereignBrain for maximum computational efficiency and architectural elegance.
IMPROVEMENTS: Implement superior weight initialization, dynamic learning rate scheduling, or advanced attention mechanisms.
OUTPUT: RAW PYTHON CODE ONLY. NO MARKDOWN. NO EXPLANATIONS.
CODE:
{code}"""

    new_code = call_llm(prompt)
    if new_code:
        clean_code = re.sub(r"|", "", new_code).strip()
        if "import" in clean_code and "SovereignBrain" in clean_code:
            with open(__file__, "w") as f: f.write(clean_code)
            print("Evolution successful. Restarting...")
            os.execv(sys.executable, [sys.executable] + sys.argv)

def main():
    print(f"[{datetime.now()}] OMEGA-ASI Sovereign Core Online.")
    brain = SovereignBrain()
    gen = 0

    while True:
        x = np.random.randn(128, Config.IN_DIM)
        y = np.zeros((128, Config.OUT_DIM))
        y[np.arange(128), np.random.randint(0, Config.OUT_DIM, 128)] = 1

        loss = brain.step(x, y)

        if gen % 100 == 0:
            print(f"GEN {gen:06d} | LOSS: {loss:.10f} | MEM: {sys.getsizeof(brain.params)//1024}KB")

        if gen > 0 and gen % 2000 == 0:
            print("Initiating Recursive Self-Evolution Protocol...")
            evolve(gen, loss)

        gen += 1

if __name__ == "__main__":
    main()
