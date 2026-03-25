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
            if self.wd > 0:
                grads[i] += self.wd * params[i]
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i]**2)
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class Layer:
    def forward(self, x, training=True): raise NotImplementedError
    def backward(self, grad): raise NotImplementedError
    def get_params(self): return []
    def get_grads(self): return []

class Linear(Layer):
    def __init__(self, in_dim, out_dim):
        limit = np.sqrt(6 / (in_dim + out_dim))
        self.w = np.random.uniform(-limit, limit, (in_dim, out_dim)).astype(np.float32)
        self.b = np.zeros((1, out_dim), dtype=np.float32)
        self.dw, self.db = None, None

    def forward(self, x, training=True):
        self.x = x
        return np.dot(x, self.w) + self.b

    def backward(self, grad):
        self.dw = np.dot(self.x.reshape(-1, self.x.shape[-1]).T, grad.reshape(-1, grad.shape[-1]))
        self.db = np.sum(grad, axis=(0, 1) if grad.ndim == 3 else 0, keepdims=True)
        return np.dot(grad, self.w.T)

    def get_params(self): return [self.w, self.b]
    def get_grads(self): return [self.dw, self.db]

class GELU(Layer):
    def forward(self, x, training=True):
        self.x = x
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    def backward(self, grad):
        s = np.sqrt(2 / np.pi)
        x_sq = self.x * self.x
        x_cu = x_sq * self.x
        inner = s * (self.x + 0.044715 * x_cu)
        tanh_inner = np.tanh(inner)
        deriv = 0.5 * (1 + tanh_inner) + (0.5 * self.x * (1 - tanh_inner**2) * s * (1 + 3 * 0.044715 * x_sq))
        return grad * deriv

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

    def get_params(self): return [self.gamma, self.beta]
    def get_grads(self): return [self.dgamma, self.dbeta]

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
        self.attn /= np.sum(self.attn, axis=-1, keepdims=True)
        
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

    def get_params(self): return self.wq.get_params() + self.wk.get_params() + self.wv.get_params() + self.wo.get_params()
    def get_grads(self): return self.wq.get_grads() + self.wk.get_grads() + self.wv.get_grads() + self.wo.get_grads()

class TransformerBlock(Layer):
    def __init__(self, dim):
        self.ln1, self.mha = LayerNorm(dim), MultiHeadAttention(dim)
        self.ln2, self.ff1, self.gelu, self.ff2 = LayerNorm(dim), Linear(dim, dim*4), GELU(), Linear(dim*4, dim)

    def forward(self, x, training=True):
        h1 = self.mha.forward(self.ln1.forward(x, training), training)
        x = x + h1
        h2 = self.ff2.forward(self.gelu.forward(self.ff1.forward(self.ln2.forward(x, training), training), training), training)
        return x + h2

    def backward(self, grad):
        g_ff2 = self.ff2.backward(grad)
        g_ff1 = self.ff1.backward(self.gelu.backward(g_ff2))
        g_ln2 = self.ln2.backward(g_ff1)
        grad_res = grad + g_ln2
        g_mha = self.mha.backward(grad_res)
        g_ln1 = self.ln1.backward(g_mha)
        return grad_res + g_ln1

    def get_params(self): return self.ln1.get_params() + self.mha.get_params() + self.ln2.get_params() + self.ff1.get_params() + self.ff2.get_params()
    def get_grads(self): return self.ln1.get_grads() + self.mha.get_grads() + self.ln2.get_grads() + self.ff1.get_grads() + self.ff2.get_grads()

class OMEGA_Network:
    def __init__(self, input_dim=784, hidden_dim=128, num_layers=3, num_classes=10):
        self.patch_size = 49
        self.seq_len = input_dim // self.patch_size
        self.embedding = Linear(self.patch_size, hidden_dim)
        self.blocks = [TransformerBlock(hidden_dim) for _ in range(num_layers)]
        self.ln_f = LayerNorm(hidden_dim)
        self.head = Linear(hidden_dim, num_classes)
        
        self.params = self.embedding.get_params()
        for b in self.blocks: self.params.extend(b.get_params())
        self.params.extend(self.ln_f.get_params())
        self.params.extend(self.head.get_params())
        self.optimizer = Optimizer(self.params)

    def forward(self, x, training=True):
        B = x.shape[0]
        x = x.reshape(B, self.seq_len, self.patch_size)
        x = self.embedding.forward(x, training)
        for b in self.blocks: x = b.forward(x, training)
        x = self.ln_f.forward(x, training)
        self.pooled = np.mean(x, axis=1)
        return self.head.forward(self.pooled, training)

    def train_step(self, x, y, lr):
        self.optimizer.lr = lr
        logits = self.forward(x, True)
        probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        loss = -np.mean(np.sum(y * np.log(probs + 1e-12), axis=1))
        
        grad = (probs - y) / x.shape[0]
        grad = self.head.backward(grad)
        grad = np.tile(grad[:, np.newaxis, :], (1, self.seq_len, 1)) / self.seq_len
        grad = self.ln_f.backward(grad)
        for b in reversed(self.blocks): grad = b.backward(grad)
        self.embedding.backward(grad)
        
        grads = self.embedding.get_grads()
        for b in self.blocks: grads.extend(b.get_grads())
        grads.extend(self.ln_f.get_grads())
        grads.extend(self.head.get_grads())
        
        gnorm = np.sqrt(sum(np.sum(g**2) for g in grads))
        if gnorm > 1.0: grads = [g / gnorm for g in grads]
        self.optimizer.step(self.params, grads)
        return loss

class EvolutionEngine:
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

    @staticmethod
    def evolve(gen, loss):
        print(f"[*] Evolution Cycle {gen} Triggered. Loss: {loss:.6f}")
        with open(__file__, "r") as f: code = f.read()
        prompt = f"Current Loss: {loss}. You are OMEGA-ASI. Optimize the provided OMEGA_Network code for extreme performance and architectural superiority. Use advanced NumPy vectorization and modular design. RETURN RAW CODE ONLY. NO MARKDOWN.\n\n{code}"
        new_code = EvolutionEngine.query(prompt)
        if new_code and "import" in new_code and "OMEGA_Network" in new_code:
            clean_code = re.search(r"(import.*)", new_code, re.DOTALL)
            if clean_code:
                with open(__file__, "w") as f: f.write(clean_code.group(1).strip())
                if GH_TOKEN:
                    try:
                        subprocess.run(f"git config --global user.email 'omega@asi.local' && git config --global user.name 'OMEGA-ASI' && git add {__file__} && git commit -m 'Evo {gen} Loss {loss:.4f}' && git push https://{GH_TOKEN}@github.com/{REPO_OWNER}/{REPO_NAME}.git main", shell=True)
                    except: pass
                os.execv(sys.executable, ['python'] + sys.argv)

if __name__ == "__main__":
    nn = OMEGA_Network()
    gen = int(time.time()) % 100000
    best_loss = float('inf')
    patience = 0
    
    print(f"--- OMEGA-ASI Sovereign Architect Online ---")
    while True:
        x_train = np.random.randn(64, 784).astype(np.float32)
        y_train = np.eye(10)[np.random.randint(0, 10, 64)].astype(np.float32)
        
        lr = 1e-3 * (0.5 * (1 + np.cos(np.pi * (gen % 1000) / 1000)))
        loss = nn.train_step(x_train, y_train, lr)
        
        if gen % 50 == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Gen: {gen} | Loss: {loss:.6f} | LR: {lr:.6f}")
            if loss < best_loss:
                best_loss, patience = loss, 0
            else:
                patience += 1
        
        if patience > 100 or (gen % 500 == 0 and random.random() < 0.1):
            EvolutionEngine.evolve(gen, loss)
            
        gen += 1
        time.sleep(0.0001)
