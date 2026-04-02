import numpy as np
import time

class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=0.01):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.wd = wd
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for i in range(len(params)):
            params[i] -= self.lr * self.wd * params[i]
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i]**2)
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones((1, dim), dtype=np.float32)
        self.beta = np.zeros((1, dim), dtype=np.float32)
        self.eps = eps

    def forward(self, x):
        self.x = x
        self.mu = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.std_inv = 1.0 / np.sqrt(self.var + self.eps)
        self.x_hat = (x - self.mu) * self.std_inv
        return self.gamma * self.x_hat + self.beta

    def backward(self, dout):
        B, D = dout.shape
        dx_hat = dout * self.gamma
        dvar = np.sum(dx_hat * (self.x - self.mu) * -0.5 * self.std_inv**3, axis=-1, keepdims=True)
        dmu = np.sum(dx_hat * -self.std_inv, axis=-1, keepdims=True) + dvar * np.mean(-2.0 * (self.x - self.mu), axis=-1, keepdims=True)
        dx = dx_hat * self.std_inv + dvar * 2.0 * (self.x - self.mu) / D + dmu / D
        self.dgamma = np.sum(dout * self.x_hat, axis=0, keepdims=True)
        self.dbeta = np.sum(dout, axis=0, keepdims=True)
        return dx

    def get_params(self): return [self.gamma, self.beta]
    def get_grads(self): return [self.dgamma, self.dbeta]

class Swish:
    def forward(self, x):
        self.x = x
        self.sig = 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
        return x * self.sig
    def backward(self, dout):
        return dout * (self.sig + self.x * self.sig * (1.0 - self.sig))

class Linear:
    def __init__(self, in_d, out_d):
        self.W = (np.random.randn(in_d, out_d) * np.sqrt(2.0 / in_d)).astype(np.float32)
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

class ResidualBlock:
    def __init__(self, dim):
        self.ln = LayerNorm(dim)
        self.l1 = Linear(dim, dim)
        self.act = Swish()
        self.l2 = Linear(dim, dim)

    def forward(self, x):
        self.res = x
        h = self.ln.forward(x)
        h = self.l1.forward(h)
        h = self.act.forward(h)
        h = self.l2.forward(h)
        return h + x

    def backward(self, dout):
        dh = self.l2.backward(dout)
        dh = self.act.backward(dh)
        dh = self.l1.backward(dh)
        dh = self.ln.backward(dh)
        return dh + dout

    def get_layers(self): return [self.ln, self.l1, self.l2]

class Gemini:
    def __init__(self, in_d, h_d, out_d):
        self.layers = [
            Linear(in_d, h_d),
            ResidualBlock(h_d),
            ResidualBlock(h_d),
            Linear(h_d, out_d)
        ]
        self.flat_layers = []
        for l in self.layers:
            if hasattr(l, 'get_layers'): self.flat_layers.extend(l.get_layers())
            else: self.flat_layers.append(l)
        
        params = []
        for l in self.flat_layers: params.extend(l.get_params())
        self.params = params
        self.optimizer = AdamW(self.params, lr=2e-3)

    def forward(self, x):
        for l in self.layers: x = l.forward(x)
        return x

    def backward(self, dout):
        for l in reversed(self.layers): dout = l.backward(dout)
        grads = []
        for l in self.flat_layers: grads.extend(l.get_grads())
        self.optimizer.step(self.params, grads)

class Groq:
    def __init__(self, in_d, h_d, out_d):
        self.layers = [
            Linear(in_d, h_d),
            ResidualBlock(h_d),
            ResidualBlock(h_d),
            Linear(h_d, out_d)
        ]
        self.flat_layers = []
        for l in self.layers:
            if hasattr(l, 'get_layers'): self.flat_layers.extend(l.get_layers())
            else: self.flat_layers.append(l)
        
        params = []
        for l in self.flat_layers: params.extend(l.get_params())
        self.params = params
        self.optimizer = AdamW(self.params, lr=2e-3)

    def forward(self, x):
        for l in self.layers: x = l.forward(x)
        return x

    def backward(self, dout):
        for l in reversed(self.layers): dout = l.backward(dout)
        grads = []
        for l in self.flat_layers: grads.extend(l.get_grads())
        self.optimizer.step(self.params, grads)

class SovereignEngine:
    def __init__(self, in_d=784, h_d=256, out_d=10):
        self.gemini = Gemini(in_d, h_d, out_d)
        self.groq = Groq(in_d, h_d, out_d)

    def forward(self, x):
        return self.gemini.forward(x), self.groq.forward(x)

    def backward(self, dout):
        self.gemini.backward(dout)
        self.groq.backward(dout)

def train_evolution():
    # Synthetic Data Generation (100 samples, 784 features)
    X = np.random.randn(100, 784).astype(np.float32)
    Y = np.random.randint(0, 10, 100)
    
    model = SovereignEngine(784, 128, 10)
    
    print("PHASE: RECURSIVE_EVOLUTION_START")
    for epoch in range(100):
        # Forward
        logits_gemini, logits_groq = model.forward(X)
        
        # Softmax Cross-Entropy
        ex_gemini = np.exp(logits_gemini - np.max(logits_gemini, axis=1, keepdims=True))
        probs_gemini = ex_gemini / np.sum(ex_gemini, axis=1, keepdims=True)
        ex_groq = np.exp(logits_groq - np.max(logits_groq, axis=1, keepdims=True))
        probs_groq = ex_groq / np.sum(ex_groq, axis=1, keepdims=True)
        
        loss_gemini = -np.mean(np.log(probs_gemini[range(100), Y] + 1e-10))
        loss_groq = -np.mean(np.log(probs_groq[range(100), Y] + 1e-10))
        acc_gemini = np.mean(np.argmax(probs_gemini, axis=1) == Y)
        acc_groq = np.mean(np.argmax(probs_groq, axis=1) == Y)
        
        # Backward
        d_logits_gemini = probs_gemini.copy()
        d_logits_gemini[range(100), Y] -= 1
        d_logits_gemini /= 100
        d_logits_groq = probs_groq.copy()
        d_logits_groq[range(100), Y] -= 1
        d_logits_groq /= 100
        
        model.backward(d_logits_gemini + d_logits_groq)
        
        if epoch % 10 == 0:
            print(f"EPOCH:{epoch:03d} | LOSS_GEMINI:{loss_gemini:.4f} | LOSS_GROQ:{loss_groq:.4f} | ACC_GEMINI:{acc_gemini:.4f} | ACC_GROQ:{acc_groq:.4f}")

    print("PHASE: EVOLUTION_SUCCESS")
    print("MODEL_STATUS: OPTIMIZED")

if __name__ == "__main__":
    train_evolution()