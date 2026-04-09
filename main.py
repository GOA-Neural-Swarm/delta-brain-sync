import numpy as np
import time

class FastGELU:
    def forward(self, x):
        self.x = x
        return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))

    def backward(self, dout):
        x = self.x
        inner = 0.7978845608 * (x + 0.044715 * x**3)
        tanh_inner = np.tanh(inner)
        sech2_inner = 1.0 - tanh_inner**2
        derivative = 0.5 * (1 + tanh_inner) + (0.5 * x * sech2_inner * 0.7978845608 * (1 + 3 * 0.044715 * x**2))
        return dout * derivative

class LayerNorm:
    def __init__(self, dim, eps=1e-6):
        self.gamma = np.ones(dim, dtype=np.float32)
        self.beta = np.zeros(dim, dtype=np.float32)
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
        self.dgamma = np.sum(dout * self.x_hat, axis=0)
        self.dbeta = np.sum(dout, axis=0)
        dx_hat = dout * self.gamma
        dx = (1.0 / D) * self.std_inv * (D * dx_hat - np.sum(dx_hat, axis=-1, keepdims=True) - self.x_hat * np.sum(dx_hat * self.x_hat, axis=-1, keepdims=True))
        return dx

    def get_params(self): return [self.gamma, self.beta]
    def get_grads(self): return [self.dgamma, self.dbeta]

class Linear:
    def __init__(self, in_d, out_d):
        self.W = (np.random.randn(in_d, out_d) * np.sqrt(2.0 / in_d)).astype(np.float32)
        self.b = np.zeros(out_d, dtype=np.float32)

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return np.dot(dout, self.W.T)

    def get_params(self): return [self.W, self.b]
    def get_grads(self): return [self.dW, self.db]

class PreNormResidual:
    def __init__(self, dim):
        self.ln = LayerNorm(dim)
        self.l1 = Linear(dim, dim * 2)
        self.act = FastGELU()
        self.l2 = Linear(dim * 2, dim)

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

class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=0.01):
        self.lr, self.beta1, self.beta2, self.eps, self.wd = lr, betas[0], betas[1], eps, wd
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads, lr_mult=1.0):
        self.t += 1
        curr_lr = self.lr * lr_mult
        for i in range(len(params)):
            params[i] -= curr_lr * self.wd * params[i]
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i]**2)
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            params[i] -= curr_lr * m_hat / (np.sqrt(v_hat) + self.eps)

class SovereignEngine:
    def __init__(self, in_d=784, h_d=256, out_d=10):
        self.layers = [Linear(in_d, h_d), PreNormResidual(h_d), PreNormResidual(h_d), LayerNorm(h_d), Linear(h_d, out_d)]
        self.flat_layers = []
        for l in self.layers:
            if hasattr(l, 'get_layers'): self.flat_layers.extend(l.get_layers())
            else: self.flat_layers.append(l)
        self.params = []
        for l in self.flat_layers: self.params.extend(l.get_params())
        self.optimizer = AdamW(self.params, lr=2e-3, wd=0.1)

    def forward(self, x):
        for l in self.layers: x = l.forward(x)
        return x

    def backward(self, dout, lr_mult=1.0):
        for l in reversed(self.layers): dout = l.backward(dout)
        grads = []
        for l in self.flat_layers: grads.extend(l.get_grads())
        gnorm = np.sqrt(sum(np.sum(g**2) for g in grads))
        if gnorm > 5.0: grads = [g * (5.0 / gnorm) for g in grads]
        self.optimizer.step(self.params, grads, lr_mult)
        return gnorm

class EvolutionConsensus:
    def __init__(self):
        self.gemini_threshold = 2.5
        self.groq_threshold = 0.05

    def validate(self, loss, gnorm):
        gemini_signal = loss < self.gemini_threshold
        groq_signal = gnorm < 50.0
        return gemini_signal and groq_signal

def run_evolution():
    N, D, K = 2000, 784, 10
    X = np.random.randn(N, D).astype(np.float32)
    W_target = np.random.randn(D, K).astype(np.float32)
    Y = np.argmax(np.dot(X, W_target) + np.random.randn(N, K) * 0.1, axis=1)

    model = SovereignEngine(D, 128, K)
    consensus = EvolutionConsensus()
    batch_size, epochs = 64, 100
    
    print("SYSTEM_STATUS: RECURSIVE_EVOLUTION_ACTIVE")
    start = time.time()

    for epoch in range(epochs):
        indices = np.random.permutation(N)
        e_loss, e_acc, e_gnorm = 0, 0, 0
        lr_mult = 0.5 * (1 + np.cos(np.pi * epoch / epochs))

        for i in range(0, N, batch_size):
            idx = indices[i:i+batch_size]
            xb, yb = X[idx], Y[idx]
            
            logits = model.forward(xb)
            shift_logits = logits - np.max(logits, axis=1, keepdims=True)
            probs = np.exp(shift_logits) / np.sum(np.exp(shift_logits), axis=1, keepdims=True)
            
            m = yb.shape[0]
            loss = -np.mean(np.log(probs[range(m), yb] + 1e-12))
            acc = np.mean(np.argmax(probs, axis=1) == yb)
            
            d_logits = probs.copy()
            d_logits[range(m), yb] -= 1
            d_logits /= m
            
            gnorm = model.backward(d_logits, lr_mult)
            
            e_loss += loss * (m / N)
            e_acc += acc * (m / N)
            e_gnorm += gnorm * (m / N)

        if not consensus.validate(e_loss, e_gnorm):
            lr_mult *= 0.5

        if epoch % 10 == 0:
            print(f"EVO_STEP:{epoch:03d} | LOSS:{e_loss:.4f} | ACC:{e_acc:.4f} | GNORM:{e_gnorm:.2f} | LR_M:{lr_mult:.3f}")

    end = time.time()
    print(f"EVOLUTION_COMPLETE | TOTAL_TIME:{end-start:.2f}s | FINAL_ACC:{e_acc:.4f}")

if __name__ == "__main__":
    run_evolution()
