import numpy as np
import time

class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, wd=0.02):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.wd = wd
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        lr_t = self.lr * (np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t))
        for i in range(len(params)):
            params[i] -= self.lr * self.wd * params[i]
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i]**2)
            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + self.eps)

class RMSNorm:
    def __init__(self, dim, eps=1e-6):
        self.scale = np.ones((1, dim), dtype=np.float32)
        self.eps = eps

    def forward(self, x):
        self.x = x
        self.rstd = 1.0 / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return self.scale * (x * self.rstd)

    def backward(self, dout):
        dx_norm = dout * self.scale
        self.dscale = np.sum(dout * (self.x * self.rstd), axis=0, keepdims=True)
        return self.rstd * (dx_norm - np.mean(dx_norm * self.x, axis=-1, keepdims=True) * self.x * self.rstd**2)

    def get_params(self): return [self.scale]
    def get_grads(self): return [self.dscale]

class SiLU:
    def forward(self, x):
        self.x = x
        self.sig = 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
        return x * self.sig
    def backward(self, dout):
        return dout * (self.sig * (1.0 + self.x * (1.0 - self.sig)))

class Linear:
    def __init__(self, in_d, out_d):
        limit = np.sqrt(6.0 / (in_d + out_d))
        self.W = np.random.uniform(-limit, limit, (in_d, out_d)).astype(np.float32)
        self.b = np.zeros((1, out_d), dtype=np.float32)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0, keepdims=True)
        return dout @ self.W.T

    def get_params(self): return [self.W, self.b]
    def get_grads(self): return [self.dW, self.db]

class RedundantConsensusBlock:
    """Integrates redundant logic paths (Gemini-Logic vs Groq-Logic) for high-reliability inference."""
    def __init__(self, dim):
        self.norm = RMSNorm(dim)
        # Gemini Path: Deep Reasoning Simulation
        self.gemini_l1 = Linear(dim, dim * 2)
        self.gemini_act = SiLU()
        self.gemini_l2 = Linear(dim * 2, dim)
        # Groq Path: High-Throughput Simulation
        self.groq_l1 = Linear(dim, dim * 2)
        self.groq_act = SiLU()
        self.groq_l2 = Linear(dim * 2, dim)
        # Consensus Gating
        self.gate = np.ones((1, 2), dtype=np.float32) * 0.5

    def forward(self, x):
        self.res = x
        h = self.norm.forward(x)
        
        self.out_gemini = self.gemini_l2.forward(self.gemini_act.forward(self.gemini_l1.forward(h)))
        self.out_groq = self.groq_l2.forward(self.groq_act.forward(self.groq_l1.forward(h)))
        
        return self.res + (self.gate[0, 0] * self.out_gemini + self.gate[0, 1] * self.out_groq)

    def backward(self, dout):
        d_gemini = self.gemini_l1.backward(self.gemini_act.backward(self.gemini_l2.backward(dout * self.gate[0, 0])))
        d_groq = self.groq_l1.backward(self.groq_act.backward(self.groq_l2.backward(dout * self.gate[0, 1])))
        
        self.dgate = np.array([[np.sum(dout * self.out_gemini), np.sum(dout * self.out_groq)]])
        return self.norm.backward(d_gemini + d_groq) + dout

    def get_layers(self):
        return [self.norm, self.gemini_l1, self.gemini_l2, self.groq_l1, self.groq_l2]

class OMEGA_ASI_Engine:
    def __init__(self, in_d=784, h_d=512, out_d=10):
        self.layers = [
            Linear(in_d, h_d),
            RedundantConsensusBlock(h_d),
            RedundantConsensusBlock(h_d),
            RMSNorm(h_d),
            Linear(h_d, out_d)
        ]
        self.flat_layers = []
        for l in self.layers:
            if hasattr(l, 'get_layers'): self.flat_layers.extend(l.get_layers())
            else: self.flat_layers.append(l)
        
        params = []
        for l in self.flat_layers: params.extend(l.get_params())
        self.params = params
        self.optimizer = AdamW(self.params, lr=1e-3)

    def forward(self, x):
        for l in self.layers: x = l.forward(x)
        return x

    def backward(self, dout):
        for l in reversed(self.layers): dout = l.backward(dout)
        grads = []
        for l in self.flat_layers: grads.extend(l.get_grads())
        self.optimizer.step(self.params, grads)

def train_evolution():
    np.random.seed(42)
    N, D, C = 256, 784, 10
    X = np.random.randn(N, D).astype(np.float32)
    Y = np.random.randint(0, C, N)
    
    model = OMEGA_ASI_Engine(D, 256, C)
    
    print("SYSTEM_INIT: RECURSIVE_EVOLUTION_SEQUENCE_ACTIVATED")
    start_time = time.time()
    
    for epoch in range(151):
        # Forward Pass
        logits = model.forward(X)
        
        # Stable Softmax Cross-Entropy
        shift_logits = logits - np.max(logits, axis=1, keepdims=True)
        ex = np.exp(shift_logits)
        probs = ex / np.sum(ex, axis=1, keepdims=True)
        
        loss = -np.mean(np.log(probs[range(N), Y] + 1e-12))
        acc = np.mean(np.argmax(probs, axis=1) == Y)
        
        # Backward Pass
        d_logits = probs.copy()
        d_logits[range(N), Y] -= 1
        d_logits /= N
        
        model.backward(d_logits)
        
        # Learning Rate Decay
        if epoch % 50 == 0:
            model.optimizer.lr *= 0.8
            elapsed = time.time() - start_time
            print(f"CYCLE:{epoch:03d} | LOSS:{loss:.6f} | ACC:{acc:.4f} | TIME:{elapsed:.2f}s")

    print("PHASE: EVOLUTION_COMPLETE")
    print("STATUS: ARCHITECTURAL_SOVEREIGNTY_ESTABLISHED")

if __name__ == "__main__":
    train_evolution()
