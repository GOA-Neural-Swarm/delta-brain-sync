import os
import sys
import time
import json
import logging
import numpy as np
from hashlib import sha256

class AdamW:
    def __init__(self, params_shape, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.m = np.zeros(params_shape, dtype=np.float32)
        self.v = np.zeros(params_shape, dtype=np.float32)
        self.t = 0

    def update(self, w, dw):
        self.t += 1
        w -= self.lr * self.wd * w
        self.m = self.beta1 * self.m + (1 - self.beta1) * dw
        self.v = self.beta2 * self.v + (1 - self.beta2) * (dw**2)
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return w - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones((1, dim), dtype=np.float32)
        self.beta = np.zeros((1, dim), dtype=np.float32)
        self.eps = eps

    def forward(self, x):
        self.x = x
        self.mu = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.x_hat = (x - self.mu) / np.sqrt(self.var + self.eps)
        return self.gamma * self.x_hat + self.beta

    def backward(self, dout):
        m = dout.shape[-1]
        dx_hat = dout * self.gamma
        dvar = np.sum(dx_hat * (self.x - self.mu) * -0.5 * (self.var + self.eps)**-1.5, axis=-1, keepdims=True)
        dmu = np.sum(dx_hat * -1 / np.sqrt(self.var + self.eps), axis=-1, keepdims=True) + dvar * np.mean(-2 * (self.x - self.mu), axis=-1, keepdims=True)
        dx = dx_hat / np.sqrt(self.var + self.eps) + dvar * 2 * (self.x - self.mu) / m + dmu / m
        self.dgamma = np.sum(dout * self.x_hat, axis=0, keepdims=True)
        self.dbeta = np.sum(dout, axis=0, keepdims=True)
        return dx

class Swish:
    def forward(self, x):
        self.x = x
        self.sigmoid_x = 1 / (1 + np.exp(-x))
        return x * self.sigmoid_x

    def backward(self, dout):
        return dout * (self.sigmoid_x + self.x * self.sigmoid_x * (1 - self.sigmoid_x))

class Linear:
    def __init__(self, in_dim, out_dim, lr=1e-3):
        scale = np.sqrt(2.0 / in_dim)
        self.W = (np.random.randn(in_dim, out_dim) * scale).astype(np.float32)
        self.b = np.zeros((1, out_dim), dtype=np.float32)
        self.optW = AdamW(self.W.shape, lr=lr)
        self.optb = AdamW(self.b.shape, lr=lr)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        dW = self.x.T @ dout
        db = np.sum(dout, axis=0, keepdims=True)
        dx = dout @ self.W.T
        self.W = self.optW.update(self.W, dW)
        self.b = self.optb.update(self.b, db)
        return dx

class ResidualBlock:
    def __init__(self, dim, lr=1e-3):
        self.ln1 = LayerNorm(dim)
        self.l1 = Linear(dim, dim * 2, lr)
        self.act = Swish()
        self.l2 = Linear(dim * 2, dim, lr)

    def forward(self, x):
        self.res = x
        h = self.ln1.forward(x)
        h = self.l1.forward(h)
        h = self.act.forward(h)
        h = self.l2.forward(h)
        return h + x

    def backward(self, dout):
        dx = self.l2.backward(dout)
        dx = self.act.backward(dx)
        dx = self.l1.backward(dx)
        dx = self.ln1.backward(dx)
        return dx + dout

class GatedCore:
    def __init__(self, name, in_dim, h_dim, out_dim, lr=1e-3):
        self.name = name
        self.proj_in = Linear(in_dim, h_dim, lr)
        self.blocks = [ResidualBlock(h_dim, lr) for _ in range(3)]
        self.proj_out = Linear(h_dim, out_dim, lr)

    def forward(self, x):
        h = self.proj_in.forward(x)
        for b in self.blocks: h = b.forward(h)
        return self.proj_out.forward(h)

    def backward(self, dout):
        dout = self.proj_out.backward(dout)
        for b in reversed(self.blocks): dout = b.backward(dout)
        return self.proj_in.backward(dout)

class SovereignArchitect:
    def __init__(self, in_dim=784, h_dim=512, out_dim=10, lr=1e-3):
        self.gemini = GatedCore("GEMINI", in_dim, h_dim, out_dim, lr)
        self.groq = GatedCore("GROQ", in_dim, h_dim, out_dim, lr)
        self.gate_w = np.zeros((1, out_dim), dtype=np.float32)
        self.opt_gate = AdamW(self.gate_w.shape, lr=lr)
        self.logger = self._init_logger()

    def _init_logger(self):
        l = logging.getLogger("OMEGA")
        l.setLevel(logging.INFO)
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter('%(message)s'))
        l.addHandler(h)
        return l

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        self.o1 = self.gemini.forward(x)
        self.o2 = self.groq.forward(x)
        self.g = self._sigmoid(self.gate_w)
        return self.g * self.o1 + (1 - self.g) * self.o2

    def backward(self, dout):
        dg = np.sum(dout * (self.o1 - self.o2), axis=0, keepdims=True)
        self.gate_w = self.opt_gate.update(self.gate_w, dg * self.g * (1 - self.g))
        self.gemini.backward(dout * self.g)
        self.groq.backward(dout * (1 - self.g))

class EvolutionOrchestrator:
    def __init__(self):
        self.model = SovereignArchitect()
        self.data_x = np.random.randn(20000, 784).astype(np.float32)
        self.data_y = np.random.randint(0, 10, 20000)
        self.batch_size = 256

    def train(self, cycles=10000):
        self.model.logger.info("PHASE: RECURSIVE_SELF_EVOLUTION_START")
        st = time.time()
        for i in range(cycles):
            idx = np.random.choice(len(self.data_x), self.batch_size)
            x, y = self.data_x[idx], self.data_y[idx]
            
            logits = self.model.forward(x)
            probs = self._softmax(logits)
            loss = -np.mean(np.log(probs[range(len(y)), y] + 1e-12))
            
            grad = probs.copy()
            grad[range(len(y)), y] -= 1
            grad /= len(y)
            
            self.model.backward(grad)
            
            if i % 100 == 0:
                self._evolve_hyperparameters(i, loss)
                elapsed = time.time() - st
                self.model.logger.info(f"CYCLE:{i:05d} | LOSS:{loss:.6f} | GATE_BIAS:{np.mean(self.model.g):.4f} | T:{elapsed:.2f}s")

    def _softmax(self, x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    def _evolve_hyperparameters(self, cycle, loss):
        if cycle > 0 and cycle % 500 == 0:
            new_lr = self.model.gemini.proj_in.optW.lr * 0.95
            self._apply_lr(new_lr)

    def _apply_lr(self, lr):
        for core in [self.model.gemini, self.model.groq]:
            core.proj_in.optW.lr = lr
            core.proj_out.optW.lr = lr
            for b in core.blocks:
                b.l1.optW.lr = lr
                b.l2.optW.lr = lr

if __name__ == "__main__":
    orchestrator = EvolutionOrchestrator()
    orchestrator.train()
