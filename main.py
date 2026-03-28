
import os
import sys
import time
import json
import logging
import numpy as np
from hashlib import sha256

class AdamWOptimizer:
    def __init__(self, shape, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.m = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)
        self.t = 0

    def step(self, w, dw):
        self.t += 1
        w -= self.lr * self.wd * w
        self.m = self.beta1 * self.m + (1 - self.beta1) * dw
        self.v = self.beta2 * self.v + (1 - self.beta2) * (dw**2)
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return w - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class BaseModule:
    def __init__(self, dims, lr=1e-3, clip=None):
        self.W1 = np.random.randn(dims[0], dims[1]).astype(np.float32) * np.sqrt(2. / dims[0])
        self.b1 = np.zeros((1, dims[1]), dtype=np.float32)
        self.W2 = np.random.randn(dims[1], dims[2]).astype(np.float32) * np.sqrt(2. / dims[1])
        self.b2 = np.zeros((1, dims[2]), dtype=np.float32)
        self.optW1 = AdamWOptimizer(self.W1.shape, lr=lr)
        self.optb1 = AdamWOptimizer(self.b1.shape, lr=lr)
        self.optW2 = AdamWOptimizer(self.W2.shape, lr=lr)
        self.optb2 = AdamWOptimizer(self.b2.shape, lr=lr)
        self.clip = clip

    def forward(self, X):
        self.x = X
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        exp_z = np.exp(self.z2 - np.max(self.z2, axis=1, keepdims=True))
        self.probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return self.probs

    def backward(self, y_true):
        m = y_true.shape[0]
        dz2 = self.probs.copy()
        dz2[range(m), y_true] -= 1
        dz2 /= m
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (self.z1 > 0)
        dW1 = np.dot(self.x.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        if self.clip:
            dW1 = np.clip(dW1, -self.clip, self.clip)
            dW2 = np.clip(dW2, -self.clip, self.clip)
        self.W2 = self.optW2.step(self.W2, dW2)
        self.b2 = self.optb2.step(self.b2, db2)
        self.W1 = self.optW1.step(self.W1, dW1)
        self.b1 = self.optb1.step(self.b1, db1)

class EvolutionEngine:
    def __init__(self, input_dim=784, hidden_dim=512, output_dim=10):
        self.gen = 0
        self.dims = (input_dim, hidden_dim, output_dim)
        self.core = BaseModule(self.dims, lr=0.001)
        self.gemini = BaseModule(self.dims, lr=0.0008, clip=1.0)
        self.groq = BaseModule(self.dims, lr=0.002, clip=1.0)
        self.ensemble_weights = np.array([0.4, 0.3, 0.3], dtype=np.float32)
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger("OMEGA-ASI")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
        logger.addHandler(handler)
        return logger

    def forward(self, X):
        p1 = self.core.forward(X)
        p2 = self.gemini.forward(X)
        p3 = self.groq.forward(X)
        return (p1 * self.ensemble_weights[0] + 
                p2 * self.ensemble_weights[1] + 
                p3 * self.ensemble_weights[2])

    def backward(self, y):
        self.core.backward(y)
        self.gemini.backward(y)
        self.groq.backward(y)

    def evolve(self, losses):
        self.gen += 1
        best_idx = np.argmin(losses)
        self.ensemble_weights[best_idx] += 0.05
        self.ensemble_weights /= self.ensemble_weights.sum()
        modules = [self.core, self.gemini, self.groq]
        winner = modules[best_idx]
        for m in modules:
            if m != winner:
                m.W1 = 0.95 * m.W1 + 0.05 * winner.W1
                m.optW1.lr *= (1.0 + np.random.uniform(-0.02, 0.02))
        self.logger.info(f"GEN {self.gen} | Best: {best_idx} | Weights: {self.ensemble_weights}")

class OmniSyncOrchestrator:
    def __init__(self):
        self.engine = EvolutionEngine()
        self.batch_size = 128
        self.data_x = np.random.randn(10000, 784).astype(np.float32)
        self.data_y = np.random.randint(0, 10, 10000)

    def run(self):
        self.engine.logger.info("Sovereign Architect Initialized.")
        while True:
            indices = np.random.choice(len(self.data_x), self.batch_size)
            x_batch, y_batch = self.data_x[indices], self.data_y[indices]
            l_core = -np.log(self.engine.core.forward(x_batch)[range(self.batch_size), y_batch] + 1e-10).mean()
            l_gemini = -np.log(self.engine.gemini.forward(x_batch)[range(self.batch_size), y_batch] + 1e-10).mean()
            l_groq = -np.log(self.engine.groq.forward(x_batch)[range(self.batch_size), y_batch] + 1e-10).mean()
            probs = self.engine.forward(x_batch)
            loss = -np.log(probs[range(self.batch_size), y_batch] + 1e-10).mean()
            self.engine.backward(y_batch)
            if self.engine.gen % 100 == 0:
                self.engine.evolve([l_core, l_gemini, l_groq])
                self.engine.logger.info(f"Loss: {loss:.6f}")
                self._snapshot()
            time.sleep(0.001)

    def _snapshot(self):
        manifest = {
            "gen": self.engine.gen,
            "hash": sha256(self.engine.core.W1.tobytes()).hexdigest()[:12],
            "weights": self.engine.ensemble_weights.tolist()
        }
        with open("integrity.jsonl", "a") as f:
            f.write(json.dumps(manifest) + "\n")

if __name__ == "__main__":
    try:
        omega = OmniSyncOrchestrator()
        omega.run()
    except KeyboardInterrupt:
        print("\n[SYSTEM] Evolution Suspended.")
        sys.exit(0)
