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
    def __init__(self, params_list, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.wd = weight_decay
        self.m = [np.zeros_like(p) for p in params_list]
        self.v = [np.zeros_like(p) for p in params_list]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        lr_t = self.lr * (np.sqrt(1 - self.betas[1]**self.t) / (1 - self.betas[0]**self.t))
        for i in range(len(params)):
            if self.wd != 0:
                params[i] -= self.lr * self.wd * params[i]
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * grads[i]
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (grads[i]**2)
            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + self.eps)

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
        # Handle 2D or 3D inputs
        x_reshaped = self.x.reshape(-1, self.x.shape[-1])
        grad_reshaped = grad.reshape(-1, grad.shape[-1])
        self.dw = np.dot(x_reshaped.T, grad_reshaped)
        self.db = np.sum(grad_reshaped, axis=0, keepdims=True)
        dx = np.dot(grad, self.w.T)
        return dx.reshape(self.x.shape)
    def get_params(self): return [self.w, self.b]
    def get_grads(self): return [self.dw, self.db]

class GELU(Layer):
    def forward(self, x, training=True):
        self.x = x
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
    def backward(self, grad):
        s = np.sqrt(2 / np.pi)
        x3 = np.power(self.x, 3)
        inner = s * (self.x + 0.044715 * x3)
        tanh_inner = np.tanh(inner)
        sech2_inner = 1.0 - tanh_inner**2
        deriv = 0.5 * (1 + tanh_inner) + (0.5 * self.x * sech2_inner * s * (1 + 3 * 0.044715 * self.x**2))
        return grad * deriv

class LayerNorm(Layer):
    def __init__(self, dim, eps=1e-6):
        self.gamma = np.ones((1, dim), dtype=np.float32)
        self.beta = np.zeros((1, dim), dtype=np.float32)
        self.eps = eps
    def forward(self, x, training=True):
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.x_hat = (x - self.mean) / np.sqrt(self.var + self.eps)
        return self.gamma * self.x_hat + self.beta
    def backward(self, grad):
        self.dgamma = np.sum(grad * self.x_hat, axis=(0, 1) if grad.ndim==3 else 0, keepdims=True)
        self.dbeta = np.sum(grad, axis=(0, 1) if grad.ndim==3 else 0, keepdims=True)
        dx_hat = grad * self.gamma
        N = grad.shape[-1]
        dx = (1. / N) * (N * dx_hat - np.sum(dx_hat, axis=-1, keepdims=True) - self.x_hat * np.sum(dx_hat * self.x_hat, axis=-1, keepdims=True)) / np.sqrt(self.var + self.eps)
        return dx
    def get_params(self): return [self.gamma, self.beta]
    def get_grads(self): return [self.dgamma, self.dbeta]

class MultiHeadAttention(Layer):
    def __init__(self, dim, heads=8):
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.wq = Linear(dim, dim)
        self.wk = Linear(dim, dim)
        self.wv = Linear(dim, dim)
        self.wo = Linear(dim, dim)
    def forward(self, x, training=True):
        B, S, D = x.shape
        self.q = self.wq.forward(x).reshape(B, S, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        self.k = self.wk.forward(x).reshape(B, S, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        self.v = self.wv.forward(x).reshape(B, S, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        
        scores = np.matmul(self.q, self.k.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        self.attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        self.attn /= np.sum(self.attn, axis=-1, keepdims=True)
        
        out = np.matmul(self.attn, self.v).transpose(0, 2, 1, 3).reshape(B, S, D)
        return self.wo.forward(out)
    def backward(self, grad):
        B, S, D = grad.shape
        g_wo = self.wo.backward(grad)
        g_attn_v = g_wo.reshape(B, S, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        
        dv = np.matmul(self.attn.transpose(0, 1, 3, 2), g_attn_v)
        d_attn = np.matmul(g_attn_v, self.v.transpose(0, 1, 3, 2))
        
        # Softmax backward
        d_scores = self.attn * (d_attn - np.sum(d_attn * self.attn, axis=-1, keepdims=True))
        d_scores /= np.sqrt(self.head_dim)
        
        dq = np.matmul(d_scores, self.k)
        dk = np.matmul(d_scores.transpose(0, 1, 3, 2), self.q)
        
        dq = dq.transpose(0, 2, 1, 3).reshape(B, S, D)
        dk = dk.transpose(0, 2, 1, 3).reshape(B, S, D)
        dv = dv.transpose(0, 2, 1, 3).reshape(B, S, D)
        
        return self.wq.backward(dq) + self.wk.backward(dk) + self.wv.backward(dv)
    def get_params(self): return self.wq.get_params() + self.wk.get_params() + self.wv.get_params() + self.wo.get_params()
    def get_grads(self): return self.wq.get_grads() + self.wk.get_grads() + self.wv.get_grads() + self.wo.get_grads()

class TransformerBlock(Layer):
    def __init__(self, dim):
        self.ln1 = LayerNorm(dim)
        self.mha = MultiHeadAttention(dim)
        self.ln2 = LayerNorm(dim)
        self.ff1 = Linear(dim, dim * 4)
        self.gelu = GELU()
        self.ff2 = Linear(dim * 4, dim)
    def forward(self, x, training=True):
        self.x_orig = x
        h1 = self.ln1.forward(x, training)
        self.attn_out = self.mha.forward(h1, training)
        x = x + self.attn_out
        self.x_mid = x
        h2 = self.ln2.forward(x, training)
        self.ff_out = self.ff2.forward(self.gelu.forward(self.ff1.forward(h2, training), training), training)
        return x + self.ff_out
    def backward(self, grad):
        g_ff2 = self.ff2.backward(grad)
        g_ff1 = self.ff1.backward(self.gelu.backward(g_ff2))
        g_ln2 = self.ln2.backward(g_ff1)
        grad = grad + g_ln2
        g_mha = self.mha.backward(grad)
        g_ln1 = self.ln1.backward(g_mha)
        return grad + g_ln1
    def get_params(self): return self.ln1.get_params() + self.mha.get_params() + self.ln2.get_params() + self.ff1.get_params() + self.ff2.get_params()
    def get_grads(self): return self.ln1.get_grads() + self.mha.get_grads() + self.ln2.get_grads() + self.ff1.get_grads() + self.ff2.get_grads()

class OMEGA_Network:
    def __init__(self):
        self.layers = [
            Linear(49, 128),
            TransformerBlock(128),
            TransformerBlock(128)
        ]
        self.head = Linear(128 * 16, 10)
        self.params = []
        for l in self.layers: self.params.extend(l.get_params())
        self.params.extend(self.head.get_params())
        self.optimizer = AdamW(self.params, lr=1e-3)

    def forward(self, x, training=True):
        x = x.reshape(-1, 16, 49)
        for layer in self.layers:
            x = layer.forward(x, training)
        self.flat_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        return self.head.forward(x, training)

    def train_step(self, x, y, lr):
        self.optimizer.lr = lr
        logits = self.forward(x, training=True)
        probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        loss = -np.mean(np.sum(y * np.log(probs + 1e-12), axis=1))
        
        grad = (probs - y) / x.shape[0]
        grad = self.head.backward(grad)
        grad = grad.reshape(self.flat_shape)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        
        grads = []
        for l in self.layers: grads.extend(l.get_grads())
        grads.extend(self.head.get_grads())
        self.optimizer.step(self.params, grads)
        return loss

class LLMRegistry:
    @staticmethod
    def query(prompt):
        if GROQ_API_KEY:
            try:
                r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}, timeout=20)
                return r.json()['choices'][0]['message']['content']
            except: pass
        if GEMINI_API_KEY:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
                r = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=20)
                return r.json()['candidates'][0]['content']['parts'][0]['text']
            except: pass
        return None

def apply_evolution(content):
    try:
        code_match = re.search(r"import.*", content, re.DOTALL)
        if code_match:
            code = code_match.group(0).strip()
            if "import" in code and "OMEGA_Network" in code:
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
        x_train = np.random.randn(64, 784).astype(np.float32)
        y_train = np.eye(10)[np.random.randint(0, 10, 64)].astype(np.float32)
        
        lr = 1e-3 * (0.5 * (1 + np.cos(np.pi * (gen % 1000) / 1000)))
        loss = nn.train_step(x_train, y_train, lr)
        
        if gen % 100 == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Gen: {gen} | Loss: {loss:.6f} | LR: {lr:.6f}")

        if random.random() < 0.001:
            print("Initiating Recursive Self-Evolution...")
            with open(__file__, "r") as f: current_code = f.read()
            prompt = f"Current loss: {loss}. Rewrite the following code to improve the OMEGA_Network. Focus on high-performance math, optimized transformer layers, or more efficient optimizers. RETURN RAW CODE ONLY. NO MARKDOWN. NO EXPLANATIONS.\n\nCODE:\n{current_code}"
            payload = LLMRegistry.query(prompt)
            if payload and apply_evolution(payload):
                print("Evolution successful. Syncing and Restarting...")
                git_sync(gen)
                os.execv(sys.executable, ['python'] + sys.argv)

        gen += 1
        time.sleep(0.001)
