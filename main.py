import numpy as np
import time

class RMSNorm:
    def __init__(self, dim, eps=1e-6):
        self.gamma = np.ones(dim, dtype=np.float32)
        self.eps = eps

    def forward(self, x):
        self.x = x
        self.norm = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return self.gamma * (x / self.norm)

    def backward(self, dout):
        x_norm = self.x / self.norm
        self.dgamma = np.sum(dout * x_norm, axis=0)
        dx_norm = dout * self.gamma
        dx = (1.0 / self.norm) * (dx_norm - x_norm * np.mean(dx_norm * x_norm, axis=-1, keepdims=True))
        return dx

    def get_params(self): return [self.gamma]
    def get_grads(self): return [self.dgamma]

class SwiGLU:
    def forward(self, x):
        self.x = x
        self.dim = x.shape[-1] // 2
        self.x1, self.x2 = x[:, :self.dim], x[:, self.dim:]
        self.sig = 1.0 / (1.0 + np.exp(-np.clip(self.x1, -20, 20)))
        self.swish = self.x1 * self.sig
        return self.swish * self.x2

    def backward(self, dout):
        dx2 = dout * self.swish
        dswish = dout * self.x2
        dx1 = dswish * (self.sig * (1.0 + self.x1 * (1.0 - self.sig)))
        return np.concatenate([dx1, dx2], axis=-1)

class Linear:
    def __init__(self, in_d, out_d, use_bias=True):
        self.W = (np.random.randn(in_d, out_d) * np.sqrt(2.0 / in_d)).astype(np.float32)
        self.b = np.zeros(out_d, dtype=np.float32) if use_bias else None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W)
        if self.b is not None: out += self.b
        return out

    def backward(self, dout):
        self.dW = np.dot(self.x.T, dout)
        if self.b is not None: self.db = np.sum(dout, axis=0)
        return np.dot(dout, self.W.T)

    def get_params(self): return [self.W, self.b] if self.b is not None else [self.W]
    def get_grads(self): return [self.dW, self.db] if self.b is not None else [self.dW]

class SovereignBlock:
    def __init__(self, dim):
        self.norm = RMSNorm(dim)
        self.proj_in = Linear(dim, dim * 4, use_bias=False)
        self.act = SwiGLU()
        self.proj_out = Linear(dim * 2, dim, use_bias=False)

    def forward(self, x):
        self.res = x
        h = self.norm.forward(x)
        h = self.proj_in.forward(h)
        h = self.act.forward(h)
        h = self.proj_out.forward(h)
        return h + self.res

    def backward(self, dout):
        dh = self.proj_out.backward(dout)
        dh = self.act.backward(dh)
        dh = self.proj_in.backward(dh)
        dh = self.norm.backward(dh)
        return dh + dout

    def get_layers(self): return [self.norm, self.proj_in, self.proj_out]

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

class ConsensusEngine:
    def __init__(self):
        self.prev_loss = float('inf')
        self.stability_count = 0

    def validate(self, loss, gnorm):
        gemini_signal = loss < self.prev_loss or loss < 2.0
        groq_signal = gnorm < 25.0
        consensus = gemini_signal and groq_signal
        self.prev_loss = loss
        return consensus, (gemini_signal, groq_signal)

class OMEGA_ARCH:
    def __init__(self, in_d=784, h_d=256, out_d=10, depth=3):
        self.layers = [Linear(in_d, h_d)]
        for _ in range(depth): self.layers.append(SovereignBlock(h_d))
        self.layers.extend([RMSNorm(h_d), Linear(h_d, out_d)])
        
        self.flat_layers = []
        for l in self.layers:
            if hasattr(l, 'get_layers'): self.flat_layers.extend(l.get_layers())
            else: self.flat_layers.append(l)
        
        self.params = []
        for l in self.flat_layers: self.params.extend(l.get_params())
        self.optimizer = AdamW(self.params, lr=3e-3, wd=0.05)

    def forward(self, x):
        for l in self.layers: x = l.forward(x)
        return x

    def backward(self, dout):
        for l in reversed(self.layers): dout = l.backward(dout)
        grads = []
        for l in self.flat_layers: grads.extend(l.get_grads())
        gnorm = np.sqrt(sum(np.sum(g**2) for g in grads))
        if gnorm > 1.0: grads = [g * (1.0 / gnorm) for g in grads]
        return grads, gnorm

def execute_evolution():
    N, D, K = 5000, 784, 10
    X = np.random.randn(N, D).astype(np.float32)
    W_true = np.random.randn(D, K).astype(np.float32)
    Y = np.argmax(np.dot(X, W_true) + 0.05 * np.random.randn(N, K), axis=1)

    model = OMEGA_ARCH(D, 160, K, depth=2)
    consensus = ConsensusEngine()
    batch_size, epochs = 128, 50

    print("STATUS: ARCHITECT_EVOLUTION_INITIATED")
    
    for epoch in range(epochs):
        idx = np.random.permutation(N)
        X, Y = X[idx], Y[idx]
        e_loss, e_acc, e_gnorm = 0, 0, 0
        lr_mult = 0.5 * (1 + np.cos(np.pi * epoch / epochs))

        for i in range(0, N, batch_size):
            xb, yb = X[i:i+batch_size], Y[i:i+batch_size]
            m = xb.shape[0]

            logits = model.forward(xb)
            probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs /= np.sum(probs, axis=1, keepdims=True)

            loss = -np.mean(np.log(probs[range(m), yb] + 1e-10))
            acc = np.mean(np.argmax(probs, axis=1) == yb)

            d_logits = probs.copy()
            d_logits[range(m), yb] -= 1
            d_logits /= m

            grads, gnorm = model.backward(d_logits)
            
            is_valid, signals = consensus.validate(loss, gnorm)
            effective_lr = lr_mult if is_valid else lr_mult * 0.1
            
            model.optimizer.step(model.params, grads, effective_lr)

            e_loss += loss * (m / N)
            e_acc += acc * (m / N)
            e_gnorm += gnorm * (m / N)

        if epoch % 5 == 0:
            print(f"EPOCH:{epoch:03d} | LOSS:{e_loss:.4f} | ACC:{e_acc:.4f} | GNORM:{e_gnorm:.3f} | CONSENSUS:{signals}")

    print(f"EVOLUTION_FINAL_ACCURACY:{e_acc:.4f}")

if __name__ == "__main__":
    execute_evolution()
