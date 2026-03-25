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

class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    def forward(self, input_data, training=True): raise NotImplementedError
    def backward(self, output_error, lr): raise NotImplementedError

class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.bias = np.zeros((1, output_size))
        self.m_w, self.v_w = np.zeros_like(self.weights), np.zeros_like(self.weights)
        self.m_b, self.v_b = np.zeros_like(self.bias), np.zeros_like(self.bias)
        self.t = 0

    def forward(self, input_data, training=True):
        self.input = input_data
        return np.dot(self.input, self.weights) + self.bias

    def backward(self, output_error, lr, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.t += 1
        grad_w = np.dot(self.input.T, output_error)
        grad_b = np.sum(output_error, axis=0, keepdims=True)
        
        self.m_w = beta1 * self.m_w + (1 - beta1) * grad_w
        self.v_w = beta2 * self.v_w + (1 - beta2) * (grad_w**2)
        self.weights -= lr * (self.m_w / (1 - beta1**self.t)) / (np.sqrt(self.v_w / (1 - beta2**self.t)) + epsilon)
        
        self.m_b = beta1 * self.m_b + (1 - beta1) * grad_b
        self.v_b = beta2 * self.v_b + (1 - beta2) * (grad_b**2)
        self.bias -= lr * (self.m_b / (1 - beta1**self.t)) / (np.sqrt(self.v_b / (1 - beta2**self.t)) + epsilon)
        return np.dot(output_error, self.weights.T)

class BatchNormalization(Layer):
    def __init__(self, dim, epsilon=1e-5, momentum=0.9):
        super().__init__()
        self.gamma = np.ones((1, dim))
        self.beta = np.zeros((1, dim))
        self.epsilon = epsilon
        self.momentum = momentum
        self.running_mean = np.zeros((1, dim))
        self.running_var = np.ones((1, dim))

    def forward(self, x, training=True):
        if training:
            mean = np.mean(x, axis=0, keepdims=True)
            var = np.var(x, axis=0, keepdims=True)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            self.x_hat = (x - mean) / np.sqrt(var + self.epsilon)
        else:
            self.x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        return self.gamma * self.x_hat + self.beta

    def backward(self, dout, lr):
        batch_size = dout.shape[0]
        dgamma = np.sum(dout * self.x_hat, axis=0, keepdims=True)
        dbeta = np.sum(dout, axis=0, keepdims=True)
        dx_hat = dout * self.gamma
        dx = (1. / batch_size) / np.sqrt(self.running_var + self.epsilon) * (
            batch_size * dx_hat - np.sum(dx_hat, axis=0) - self.x_hat * np.sum(dx_hat * self.x_hat, axis=0)
        )
        self.gamma -= lr * dgamma
        self.beta -= lr * dbeta
        return dx

class Activation(Layer):
    def __init__(self, func, func_prime):
        super().__init__()
        self.func, self.func_prime = func, func_prime
    def forward(self, x, training=True):
        self.input = x
        return self.func(x)
    def backward(self, dout, lr):
        return self.func_prime(self.input) * dout

def relu(x): return np.maximum(0, x)
def relu_prime(x): return (x > 0).astype(float)
def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

class OMEGA_Network:
    def __init__(self):
        self.layers = [
            Dense(784, 512), BatchNormalization(512), Activation(relu, relu_prime),
            Dense(512, 256), BatchNormalization(256), Activation(relu, relu_prime),
            Dense(256, 10)
        ]

    def predict(self, x, training=False):
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return softmax(x)

    def train_step(self, x, y, lr):
        logits = self.predict(x, training=True)
        loss = -np.mean(np.sum(y * np.log(logits + 1e-12), axis=1))
        error = (logits - y) / x.shape[0]
        for layer in reversed(self.layers):
            error = layer.backward(error, lr)
        return loss

class LLMRegistry:
    @staticmethod
    def query(prompt):
        # Redundant Logic: Groq Primary, Gemini Fallback
        if GROQ_API_KEY:
            try:
                resp = requests.post("https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "temperature": 0.1},
                    timeout=15)
                return resp.json()['choices'][0]['message']['content']
            except: pass
        if GEMINI_API_KEY:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
                resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=15)
                return resp.json()['candidates'][0]['content']['parts'][0]['text']
            except: pass
        return None

def apply_evolution(content):
    try:
        code = re.search(r"import.*", content, re.DOTALL).group(0) if "import" in content else content
        code = code.split("")[-1].split("")[0].strip() if "" in code else code
        if "OMEGA_Network" in code:
            with open("main.py", "w") as f: f.write(code)
            return True
    except: pass
    return False

def git_sync(gen):
    if not GH_TOKEN: return
    cmd = f"git config --global user.email 'omega@asi.local' && git config --global user.name 'OMEGA-ASI' && git add main.py && git commit -m 'Evo {gen}' && git push https://{GH_TOKEN}@github.com/{REPO_OWNER}/{REPO_NAME}.git main"
    subprocess.run(cmd, shell=True, capture_output=True)

if __name__ == "__main__":
    nn = OMEGA_Network()
    gen = int(time.time()) % 100000
    print(f"OMEGA-ASI Sovereign Architect Online. Gen: {gen}")

    while True:
        x_train = np.random.randn(128, 784)
        y_train = np.eye(10)[np.random.randint(0, 10, 128)]
        
        lr = 0.001 * (0.5 * (1 + np.cos(np.pi * (gen % 100) / 100))) # Cosine Annealing
        losses = [nn.train_step(x_train, y_train, lr) for _ in range(20)]
        avg_loss = np.mean(losses)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Gen: {gen} | Loss: {avg_loss:.8f} | LR: {lr:.6f}")

        if random.random() > 0.9:
            print("Initiating Recursive Self-Evolution...")
            prompt = f"Rewrite the entire main.py. Current Loss: {avg_loss}. Improve OMEGA_Network with Residual connections or Attention mechanisms. Return ONLY code."
            payload = LLMRegistry.query(prompt)
            if payload and apply_evolution(payload):
                git_sync(gen)
                os.execv(sys.executable, ['python'] + sys.argv)

        time.sleep(1)
        gen += 1
