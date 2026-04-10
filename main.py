import numpy as np
import time

class GELU:
    def forward(self, x):
        self.x = x
        return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))

    def backward(self, dout):
        x = self.x
        tanh_part = np.tanh(0.7978845608 * (x + 0.044715 * x**3))
        grad = 0.5 * (1 + tanh_part) + 0.5 * x * (1 - tanh_part**2) * 0.7978845608 * (1 + 3 * 0.044715 * x**2)
        return dout * grad

class RMSNorm:
    def __init__(self, dim, eps=1e-6):
        self.scale = np.ones(dim, dtype=np.float32)
        self.eps = eps

    def forward(self, x):
        self.x = x
        self.rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        self.x_normed = x / self.rms
        return self.scale * self.x_normed

    def backward(self, dout):
        self.dscale = np.sum(dout * self.x_normed, axis=0)
        dx_normed = dout * self.scale
        N = dout.shape[-1]
        dx = (1.0 / self.rms) * (dx_normed - self.x_normed * np.mean(dx_normed * self.x_normed, axis=-1, keepdims=True))
        return dx

    def get_params(self): return [self.scale]
    def get_grads(self): return [self.dscale]

class Linear:
    def __init__(self, in_d, out_d, use_bias=True):
        limit = np.sqrt(6.0 / (in_d + out_d))
        self.W = np.random.uniform(-limit, limit, (in_d, out_d)).astype(np.float32)
        self.use_bias = use_bias
        if use_bias:
            self.b = np.zeros(out_d, dtype=np.float32)

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W)
        if self.use_bias:
            out += self.b
        return out

    def backward(self, dout):
        self.dW = np.dot(self.x.T, dout)
        if self.use_bias:
            self.db = np.sum(dout, axis=0)
        dx = np.dot(dout, self.W.T)
        return dx

    def get_params(self): return [self.W, self.b] if self.use_bias else [self.W]
    def get_grads(self): return [self.dW, self.db] if self.use_bias else [self.dW]

class DualPathEngine:
    """Consensus Logic: Gemini (Stability) + Groq (Throughput)"""
    def __init__(self, dim):
        self.gemini_proj = Linear(dim, dim)
        self.groq_proj = Linear(dim, dim)
        self.gate = Linear(dim, 2)
        
    def forward(self, x):
        self.x = x
        self.out_gemini = self.gemini_proj.forward(x)
        self.out_groq = self.groq_proj.forward(x)
        
        g_logits = self.gate.forward(x)
        g_exp = np.exp(g_logits - np.max(g_logits, axis=-1, keepdims=True))
        self.probs = g_exp / np.sum(g_exp, axis=-1, keepdims=True)
        
        return self.probs[:, 0:1] * self.out_gemini + self.probs[:, 1:2] * self.out_groq

    def backward(self, dout):
        d_gemini = dout * self.probs[:, 0:1]
        d_groq = dout * self.probs[:, 1:2]
        
        d_p0 = np.sum(dout * self.out_gemini, axis=-1, keepdims=True)
        d_p1 = np.sum(dout * self.out_groq, axis=-1, keepdims=True)
        d_probs_raw = np.concatenate([d_p0, d_p1], axis=-1)
        
        d_gate_logits = self.probs * (d_probs_raw - np.sum(self.probs * d_probs_raw, axis=-1, keepdims=True))
        
        dx_gate = self.gate.backward(d_gate_logits)
        dx_gemini = self.gemini_proj.backward(d_gemini)
        dx_groq = self.groq_proj.backward(d_groq)
        
        return dx_gate + dx_gemini + dx_groq

    def get_params(self):
        return self.gemini_proj.get_params() + self.groq_proj.get_params() + self.gate.get_params()

    def get_grads(self):
        return self.gemini_proj.get_grads() + self.groq_proj.get_grads() + self.gate.get_grads()

class SovereignBlock:
    def __init__(self, dim, expansion=4):
        self.norm1 = RMSNorm(dim)
        self.engine = DualPathEngine(dim)
        self.norm2 = RMSNorm(dim)
        self.ff1 = Linear(dim, expansion * dim)
        self.act = GELU()
        self.ff2 = Linear(expansion * dim, dim)

    def forward(self, x):
        self.res1 = x
        h = self.norm1.forward(x)
        h = self.engine.forward(h)
        x = self.res1 + h
        
        self.res2 = x
        h = self.norm2.forward(x)
        h = self.ff1.forward(h)
        h = self.act.forward(h)
        h = self.ff2.forward(h)
        return self.res2 + h

    def backward(self, dout):
        d_res2 = dout
        dh = self.ff2.backward(dout)
        dh = self.act.backward(dh)
        dh = self.ff1.backward(dh)
        dh = self.norm2.backward(dh)
        dout = d_res2 + dh
        
        d_res1 = dout
        dh = self.engine.backward(self.norm1.forward(self.res1))
        dh = self.norm1.backward(dh)
        return d_res1 + dh

    def get_params(self):
        return self.norm1.get_params() + self.engine.get_params() + self.norm2.get_params() + self.ff1.get_params() + self.ff2.get_params()

    def get_grads(self):
        return self.norm1.get_grads() + self.engine.get_grads() + self.norm2.get_grads() + self.ff1.get_grads() + self.ff2.get_grads()

class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=0.01):
        self.lr, self.beta1, self.beta2, self.eps, self.wd = lr, betas[0], betas[1], eps, wd
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for i in range(len(params)):
            p, g = params[i], grads[i]
            p -= self.lr * self.wd * p
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g**2)
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class SovereignArchitect:
    def __init__(self, in_d, h_d, out_d, depth=2):
        self.stem = Linear(in_d, h_d)
        self.blocks = [SovereignBlock(h_d) for _ in range(depth)]
        self.norm = RMSNorm(h_d)
        self.head = Linear(h_d, out_d)
        
        self.params = self.stem.get_params()
        for b in self.blocks: self.params.extend(b.get_params())
        self.params.extend(self.norm.get_params() + self.head.get_params())
        self.optimizer = AdamW(self.params, lr=1e-3, wd=0.05)

    def forward(self, x):
        x = self.stem.forward(x)
        for b in self.blocks: x = b.forward(x)
        x = self.norm.forward(x)
        return self.head.forward(x)

    def backward(self, dout):
        dout = self.head.backward(dout)
        dout = self.norm.backward(dout)
        for b in reversed(self.blocks): dout = b.backward(dout)
        self.stem.backward(dout)
        
        grads = self.stem.get_grads()
        for b in self.blocks: grads.extend(b.get_grads())
        grads.extend(self.norm.get_grads() + self.head.get_grads())
        return grads

def train():
    N, D, K = 5000, 784, 10
    X = np.random.randn(N, D).astype(np.float32)
    y = np.random.randint(0, K, N)
    
    model = SovereignArchitect(D, 128, K, depth=2)
    batch_size = 128
    epochs = 50
    
    print("OMEGA-ASI | RECURSIVE EVOLUTION INITIATED")
    
    for epoch in range(epochs):
        idx = np.random.permutation(N)
        epoch_loss, epoch_acc = 0, 0
        
        for i in range(0, N, batch_size):
            batch_idx = idx[i:i+batch_size]
            xb, yb = X[batch_idx], y[batch_idx]
            m = xb.shape[0]
            
            logits = model.forward(xb)
            probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs /= np.sum(probs, axis=1, keepdims=True)
            
            loss = -np.mean(np.log(probs[range(m), yb] + 1e-10))
            acc = np.mean(np.argmax(probs, axis=1) == yb)
            
            d_logits = (probs.copy())
            d_logits[range(m), yb] -= 1
            d_logits /= m
            
            grads = model.backward(d_logits)
            
            # Gradient Clipping
            gnorm = np.sqrt(sum(np.sum(g**2) for g in grads))
            if gnorm > 1.0:
                grads = [g * (1.0 / gnorm) for g in grads]
                
            model.optimizer.step(model.params, grads)
            
            epoch_loss += loss * (m / N)
            epoch_acc += acc * (m / N)
            
        if epoch % 5 == 0:
            print(f"EPOCH:{epoch:03d} | LOSS:{epoch_loss:.4f} | ACC:{epoch_acc:.4f}")

if __name__ == "__main__":
    train()
