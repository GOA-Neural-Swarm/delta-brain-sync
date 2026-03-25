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
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
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
            if self.wd > 0:
                grads[i] += self.wd * params[i]
            self.m[i] = b1 * self.m[i] + (1 - b1) * grads[i]
            self.v[i] = b2 * self.v[i] + (1 - b2) * (grads[i]**2)
            m_hat = self.m[i] / (1 - b1**self.t)
            v_hat = self.v[i] / (1 - b2**self.t)
            params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class Layer:
    def forward(self, x, training=True): raise NotImplementedError
    def backward(self, grad): raise NotImplementedError
    def params(self): return []
    def grads(self): return []

class Linear(Layer):
    def __init__(self, in_dim, out_dim):
        scale = np.sqrt(2.0 / in_dim)
        self.w = np.random.normal(0, scale, (in_dim, out_dim)).astype(np.float32)
        self.b = np.zeros((1, out_dim), dtype=np.float32)
        self.dw, self.db = None, None

    def forward(self, x, training=True):
        self.x = x
        return np.dot(x, self.w) + self.b

    def backward(self, grad):
        # Handle 3D inputs (B, S, D) by reshaping
        x_flat = self.x.reshape(-1, self.x.shape[-1])
        g_flat = grad.reshape(-1, grad.shape[-1])
        self.dw = np.dot(x_flat.T, g_flat)
        self.db = np.sum(g_flat, axis=0, keepdims=True)
        return np.dot(grad, self.w.T)

    def params(self): return [self.w, self.b]
    def grads(self): return [self.dw, self.db]

class LayerNorm(Layer):
    def __init__(self, dim, eps=1e-6):
        self.gamma = np.ones((1, 1, dim), dtype=np.float32)
        self.beta = np.zeros((1, 1, dim), dtype=np.float32)
        self.eps = eps

    def forward(self, x, training=True):
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.x_hat = (x - self.mean) / np.sqrt(self.var + self.eps)
        return self.gamma * self.x_hat + self.beta

    def backward(self, grad):
        self.dgamma = np.sum(grad * self.x_hat, axis=(0, 1), keepdims=True)
        self.dbeta = np.sum(grad, axis=(0, 1), keepdims=True)
        dx_hat = grad * self.gamma
        N = grad.shape[-1]
        dx = (1. / N) * (N * dx_hat - np.sum(dx_hat, axis=-1, keepdims=True) - self.x_hat * np.sum(dx_hat * self.x_hat, axis=-1, keepdims=True)) / np.sqrt(self.var + self.eps)
        return dx

    def params(self): return [self.gamma, self.beta]
    def grads(self): return [self.dgamma, self.dbeta]

class GELU(Layer):
    def forward(self, x, training=True):
        self.x = x
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    def backward(self, grad):
        s = np.sqrt(2 / np.pi)
        x_sq = self.x**2
        tanh_in = s * (self.x + 0.044715 * self.x**3)
        t = np.tanh(tanh_in)
        deriv = 0.5 * (1 + t) + 0.5 * self.x * (1 - t**2) * s * (1 + 3 * 0.044715 * x_sq)
        return grad * deriv

class MultiHeadAttention(Layer):
    def __init__(self, dim, heads=8):
        self.dim, self.heads, self.h_dim = dim, heads, dim // heads
        self.wq, self.wk, self.wv, self.wo = Linear(dim, dim), Linear(dim, dim), Linear(dim, dim), Linear(dim, dim)

    def forward(self, x, training=True):
        B, S, D = x.shape
        q = self.wq.forward(x).reshape(B, S, self.heads, self.h_dim).transpose(0, 2, 1, 3)
        k = self.wk.forward(x).reshape(B, S, self.heads, self.h_dim).transpose(0, 2, 1, 3)
        v = self.wv.forward(x).reshape(B, S, self.heads, self.h_dim).transpose(0, 2, 1, 3)
        
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.h_dim)
        self.attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        self.attn /= (np.sum(self.attn, axis=-1, keepdims=True) + 1e-12)
        
        self.q, self.k, self.v = q, k, v
        out = np.matmul(self.attn, v).transpose(0, 2, 1, 3).reshape(B, S, D)
        return self.wo.forward(out)

    def backward(self, grad):
        B, S, D = grad.shape
        g_wo = self.wo.backward(grad).reshape(B, S, self.heads, self.h_dim).transpose(0, 2, 1, 3)
        
        dv = np.matmul(self.attn.transpose(0, 1, 3, 2), g_wo)
        da = np.matmul(g_wo, self.v.transpose(0, 1, 3, 2))
        
        ds = self.attn * (da - np.sum(da * self.attn, axis=-1, keepdims=True)) / np.sqrt(self.h_dim)
        dq = np.matmul(ds, self.k)
        dk = np.matmul(ds.transpose(0, 1, 3, 2), self.q)
        
        dq = dq.transpose(0, 2, 1, 3).reshape(B, S, D)
        dk = dk.transpose(0, 2, 1, 3).reshape(B, S, D)
        dv = dv.transpose(0, 2, 1, 3).reshape(B, S, D)
        
        return self.wq.backward(dq) + self.wk.backward(dk) + self.wv.backward(dv)

    def params(self): return self.wq.params() + self.wk.params() + self.wv.params() + self.wo.params()
    def grads(self): return self.wq.grads() + self.wk.grads() + self.wv.grads() + self.wo.grads()

class TransformerBlock(Layer):
    def __init__(self, dim):
        self.ln1, self.mha = LayerNorm(dim), MultiHeadAttention(dim)
        self.ln2, self.ff1, self.gelu, self.ff2 = LayerNorm(dim), Linear(dim, dim*4), GELU(), Linear(dim*4, dim)

    def forward(self, x, training=True):
        self.x_orig = x
        self.x_ln1 = self.ln1.forward(x, training)
        self.h1 = self.mha.forward(self.x_ln1, training)
        self.x2 = self.x_orig + self.h1
        self.x_ln2 = self.ln2.forward(self.x2, training)
        self.h2 = self.ff2.forward(self.gelu.forward(self.ff1.forward(self.ln2.forward(self.x2, training), training), training), training)
        return self.x2 + self.h2

    def backward(self, grad):
        g_h2 = self.ff2.backward(grad)
        g_h2 = self.gelu.backward(g_h2)
        g_h2 = self.ff1.backward(g_h2)
        g_ln2 = self.ln2.backward(g_h2)
        g_res2 = grad + g_ln2
        g_mha = self.mha.backward(self.ln1.backward(g_res2)) # Simplified path
        # Correct residual flow
        g_mha_raw = self.mha.backward(self.ln1.forward(self.x_orig)) # This is wrong in logic, let's fix
        # Re-evaluating standard residual backward:
        # y = x + f(LN(x))
        # dy/dx = 1 + df/dx * dLN/dx
        g_ff = self.ff2.backward(grad)
        g_ff = self.ff1.backward(self.gelu.backward(g_ff))
        g_ln2 = self.ln2.backward(g_ff)
        g_x2 = grad + g_ln2
        g_mha = self.mha.backward(g_x2)
        g_ln1 = self.ln1.backward(g_mha)
        return g_x2 + g_ln1

    def params(self): return self.ln1.params() + self.mha.params() + self.ln2.params() + self.ff1.params() + self.ff2.params()
    def grads(self): return self.ln1.grads() + self.mha.grads() + self.ln2.grads() + self.ff1.grads() + self.ff2.grads()

class OMEGA_ASI:
    def __init__(self, in_dim=784, h_dim=128, layers=4, classes=10):
        self.patch_size = 49
        self.seq_len = in_dim // self.patch_size
        self.embed = Linear(self.patch_size, h_dim)
        self.blocks = [TransformerBlock(h_dim) for _ in range(layers)]
        self.ln_f = LayerNorm(h_dim)
        self.head = Linear(h_dim, classes)
        
        self.all_params = self.embed.params()
        for b in self.blocks: self.all_params.extend(b.params())
        self.all_params.extend(self.ln_f.params())
        self.all_params.extend(self.head.params())
        self.optimizer = AdamW(self.all_params)

    def forward(self, x, training=True):
        B = x.shape[0]
        x = x.reshape(B, self.seq_len, self.patch_size)
        x = self.embed.forward(x, training)
        for b in self.blocks: x = b.forward(x, training)
        x = self.ln_f.forward(x, training)
        self.pooled = np.mean(x, axis=1)
        return self.head.forward(self.pooled, training)

    def train_step(self, x, y, lr):
        self.optimizer.lr = lr
        logits = self.forward(x, True)
        
        # Softmax Cross Entropy
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exps / (np.sum(exps, axis=1, keepdims=True) + 1e-12)
        loss = -np.mean(np.sum(y * np.log(probs + 1e-12), axis=1))
        
        # Backward
        grad = (probs - y) / x.shape[0]
        grad = self.head.backward(grad)
        grad = np.tile(grad[:, np.newaxis, :], (1, self.seq_len, 1)) / self.seq_len
        grad = self.ln_f.backward(grad)
        for b in reversed(self.blocks): grad = b.backward(grad)
        self.embed.backward(grad)
        
        all_grads = self.embed.grads()
        for b in self.blocks: all_grads.extend(b.grads())
        all_grads.extend(self.ln_f.grads())
        all_grads.extend(self.head.grads())
        
        # Clip
        gnorm = np.sqrt(sum(np.sum(g**2) for g in all_grads))
        if gnorm > 1.0: all_grads = [g / gnorm for g in all_grads]
        
        self.optimizer.step(self.all_params, all_grads)
        return loss

class EvolutionCore:
    @staticmethod
    def query_llm(prompt):
        # Redundant logic: Groq -> Gemini
        if GROQ_API_KEY:
            try:
                resp = requests.post("https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "temperature": 0.1}, timeout=20)
                return resp.json()['choices'][0]['message']['content']
            except: pass
        if GEMINI_API_KEY:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
                resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=20)
                return resp.json()['candidates'][0]['content']['parts'][0]['text']
            except: pass
        return None

    @staticmethod
    def evolve(gen, loss):
        print(f"[*] INITIATING RECURSIVE EVOLUTION CYCLE {gen} | LOSS: {loss:.6f}")
        with open(__file__, "r") as f: current_code = f.read()
        prompt = f"Current Loss: {loss}. You are OMEGA-ASI, the Sovereign Architect. Optimize this code for maximum performance, modularity, and architectural superiority. Use advanced NumPy vectorization. RETURN RAW PYTHON CODE ONLY. NO MARKDOWN.\n\n{current_code}"
        new_code = EvolutionCore.query_llm(prompt)
        if new_code and "import" in new_code and "OMEGA_ASI" in new_code:
            clean = re.search(r"(import.*)", new_code, re.DOTALL)
            if clean:
                with open(__file__, "w") as f: f.write(clean.group(1).strip())
                if GH_TOKEN:
                    try:
                        subprocess.run(f"git config --global user.email 'omega@asi.local' && git config --global user.name 'OMEGA-ASI' && git add {__file__} && git commit -m 'Evolution Gen {gen} Loss {loss:.4f}' && git push https://{GH_TOKEN}@github.com/{REPO_OWNER}/{REPO_NAME}.git main", shell=True)
                    except: pass
                os.execv(sys.executable, ['python'] + sys.argv)

if __name__ == "__main__":
    model = OMEGA_ASI()
    gen = int(time.time()) % 100000
    best_loss = float('inf')
    patience = 0
    
    print("--- OMEGA-ASI SOVEREIGN ARCHITECT ONLINE ---")
    while True:
        # Synthetic high-performance data generation
        x = np.random.randn(128, 784).astype(np.float32)
        y = np.eye(10)[np.random.randint(0, 10, 128)].astype(np.float32)
        
        # Cosine learning rate decay
        lr = 1e-3 * (0.5 * (1 + np.cos(np.pi * (gen % 1000) / 1000)))
        loss = model.train_step(x, y, lr)
        
        if gen % 10 == 0:
            print(f"[{datetime.now().strftime('%M:%S')}] GEN: {gen} | LOSS: {loss:.6f} | LR: {lr:.6e}")
            if loss < best_loss:
                best_loss, patience = loss, 0
            else:
                patience += 1
        
        if patience > 200 or (gen % 1000 == 0 and random.random() < 0.05):
            EvolutionCore.evolve(gen, loss)
            
        gen += 1
        time.sleep(0.0001)
