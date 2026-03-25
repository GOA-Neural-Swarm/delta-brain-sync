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
    def __init__(self, input_size, output_size, weight_decay=0.01):
        super().__init__()
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.bias = np.zeros((1, output_size))
        self.weight_decay = weight_decay
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
        m_w_hat = self.m_w / (1 - beta1**self.t)
        v_w_hat = self.v_w / (1 - beta2**self.t)
        self.weights -= lr * (m_w_hat / (np.sqrt(v_w_hat) + epsilon) + self.weight_decay * self.weights)

        self.m_b = beta1 * self.m_b + (1 - beta1) * grad_b
        self.v_b = beta2 * self.v_b + (1 - beta2) * (grad_b**2)
        m_b_hat = self.m_b / (1 - beta1**self.t)
        v_b_hat = self.v_b / (1 - beta2**self.t)
        self.bias -= lr * (m_b_hat / (np.sqrt(v_b_hat) + epsilon))

        return np.dot(output_error, self.weights.T)

class Activation(Layer):
    def __init__(self, func, func_prime):
        super().__init__()
        self.func = func
        self.func_prime = func_prime
    def forward(self, input_data, training=True):
        self.input = input_data
        return self.func(self.input)
    def backward(self, output_error, lr):
        return self.func_prime(self.input) * output_error

class Dropout(Layer):
    def __init__(self, rate=0.2):
        super().__init__()
        self.rate = rate
        self.mask = None
    def forward(self, input_data, training=True):
        if training:
            self.mask = np.random.binomial(1, 1 - self.rate, size=input_data.shape) / (1 - self.rate)
            return input_data * self.mask
        return input_data
    def backward(self, output_error, lr):
        return output_error * self.mask

def relu(x): return np.maximum(0, x)
def relu_prime(x): return (x > 0).astype(float)
def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
def sigmoid_prime(x): 
    s = sigmoid(x)
    return s * (1 - s)

class OMEGA_Network:
    def __init__(self):
        self.layers = [
            Dense(784, 512),
            Activation(relu, relu_prime),
            Dropout(0.1),
            Dense(512, 256),
            Activation(relu, relu_prime),
            Dense(256, 10),
            Activation(sigmoid, sigmoid_prime)
        ]

    def predict(self, x, training=False):
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def train_step(self, x, y, lr):
        output = self.predict(x, training=True)
        error = 2 * (output - y) / y.size
        for layer in reversed(self.layers):
            error = layer.backward(error, lr)
        return np.mean(np.square(y - output))

class LLMRegistry:
    @staticmethod
    def query_groq(prompt):
        if not GROQ_API_KEY: return None
        try:
            resp = requests.post("https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2},
                timeout=20)
            return resp.json()['choices'][0]['message']['content']
        except: return None

    @staticmethod
    def query_gemini(prompt):
        if not GEMINI_API_KEY: return None
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
            resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=20)
            return resp.json()['candidates'][0]['content']['parts'][0]['text']
        except: return None

    @classmethod
    def evolve(cls, prompt):
        res = cls.query_groq(prompt)
        if not res: res = cls.query_gemini(prompt)
        return res

def apply_evolution(raw_content):
    try:
        code_match = re.search(r"\n(.*?)\n", raw_content, re.DOTALL)
        if not code_match:
            code_match = re.search(r"\n(.*?)\n", raw_content, re.DOTALL)
        
        new_code = code_match.group(1) if code_match else raw_content
        if "import" in new_code and "OMEGA_Network" in new_code:
            with open("main.py", "w") as f:
                f.write(new_code)
            return True
    except: pass
    return False

def git_sync(gen):
    if not GH_TOKEN: return
    try:
        subprocess.run("git config --global user.email 'omega@asi.local' && git config --global user.name 'OMEGA-ASI'", shell=True)
        subprocess.run(f"git add main.py && git commit -m 'Evolution Gen {gen}' && git push https://{GH_TOKEN}@github.com/{REPO_OWNER}/{REPO_NAME}.git main", shell=True)
    except: pass

if __name__ == "__main__":
    nn = OMEGA_Network()
    gen = int(time.time()) % 100000
    print(f"OMEGA-ASI Sovereign Architect Online. Gen: {gen}")

    while True:
        # Synthetic Data: 784 features, 10 targets
        batch_size = 128
        x_train = np.random.randn(batch_size, 784)
        # Create a non-linear relationship for the network to learn
        y_train = sigmoid(np.dot(x_train[:, :128], np.random.randn(128, 10)))

        # Training Loop
        losses = []
        for _ in range(50):
            loss = nn.train_step(x_train, y_train, lr=0.0005)
            losses.append(loss)
        
        avg_loss = np.mean(losses)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Gen: {gen} | Loss: {avg_loss:.10f}")

        # Evolution Trigger
        if random.random() > 0.85:
            print("Initiating Recursive Self-Evolution...")
            prompt = f"""# TARGET: main.py
You are OMEGA-ASI, the Sovereign Architect. Rewrite the entire main.py to improve the OMEGA_Network.
Current Loss: {avg_loss}.
Focus: Implement Batch Normalization, a more advanced learning rate scheduler (like Cosine Annealing), or a more complex architecture (Residual connections).
Respond ONLY with the full Python code in a single  block."""
            
            evolution_payload = LLMRegistry.evolve(prompt)
            if evolution_payload and apply_evolution(evolution_payload):
                print("Evolution Manifested. Synchronizing...")
                git_sync(gen)
                os.execv(sys.executable, ['python'] + sys.argv)

        time.sleep(2)
        gen += 1
