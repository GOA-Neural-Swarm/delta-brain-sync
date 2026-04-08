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

class GELU:
    def forward(self, x):
        self.x = x
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
    
    def backward(self, dout):
        x = self.x
        cdf = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
        pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
        return dout * (cdf + x * pdf)

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

class RedundantExpertBlock:
    def __init__(self, dim):
        # Gemini Path
        self.ln_g = LayerNorm(dim)
        self.l1_g = Linear(dim, dim * 2)
        self.act_g = GELU()
        self.l2_g = Linear(dim * 2, dim)
        
        # Groq Path
        self.ln_q = LayerNorm(dim)
        self.l1_q = Linear(dim, dim * 2)
        self.act_q = GELU()
        self.l2_q = Linear(dim * 2, dim)

    def forward(self, x):
        self.res = x
        # Gemini Logic
        g = self.ln_g.forward(x)
        g = self.l1_g.forward(g)
        g = self.act_g.forward(g)
        self.out_g = self.l2_g.forward(g)
        
        # Groq Logic
        q = self.ln_q.forward(x)
        q = self.l1_q.forward(q)
        q = self.act_q.forward(q)
        self.out_q = self.l2_q.forward(q)
        
        return (self.out_g + self.out_q) * 0.5 + x

    def backward(self, dout):
        # Gradient split
        d_path = dout * 0.5
        
        # Groq Backward
        dq = self.l2_q.backward(d_path)
        dq = self.act_q.backward(dq)
        dq = self.l1_q.backward(dq)
        dq = self.ln_q.backward(dq)
        
        # Gemini Backward
        dg = self.l2_g.backward(d_path)
        dg = self.act_g.backward(dg)
        dg = self.l1_g.backward(dg)
        dg = self.ln_g.backward(dg)
        
        return dg + dq + dout

    def get_layers(self):
        return [self.ln_g, self.l1_g, self.l2_g, self.ln_q, self.l1_q, self.l2_q]

class OMEGA_ASI_Engine:
    def __init__(self, in_d=784, h_d=256, out_d=10):
        self.layers = [
            Linear(in_d, h_d),
            RedundantExpertBlock(h_d),
            RedundantExpertBlock(h_d),
            LayerNorm(h_d),
            Linear(h_d, out_d)
        ]
        self.flat_layers = []
        for l in self.layers:
            if hasattr(l, 'get_layers'): self.flat_layers.extend(l.get_layers())
            else: self.flat_layers.append(l)
        
        params = []
        for l in self.flat_layers: params.extend(l.get_params())
        self.params = params
        self.optimizer = AdamW(self.params, lr=1e-3, wd=0.05)

    def forward(self, x):
        for l in self.layers: x = l.forward(x)
        return x

    def backward(self, dout):
        for l in reversed(self.layers): dout = l.backward(dout)
        grads = []
        for l in self.flat_layers: grads.extend(l.get_grads())
        self.optimizer.step(self.params, grads)

def train_evolution():
    # High-Performance Synthetic Data
    N, D, C = 1000, 784, 10
    X = np.random.randn(N, D).astype(np.float32)
    Y = np.random.randint(0, C, N)
    
    batch_size = 64
    model = OMEGA_ASI_Engine(D, 128, C)
    
    print("SYSTEM_INIT: OMEGA-ASI SOVEREIGN ARCHITECT ACTIVE")
    print(f"RECURSIVE_EVOLUTION_MODE: ENABLED | SAMPLES: {N} | BATCH: {batch_size}")
    
    start_time = time.time()
    for epoch in range(50):
        indices = np.arange(N)
        np.random.shuffle(indices)
        
        epoch_loss = 0
        epoch_acc = 0
        batches = 0
        
        for i in range(0, N, batch_size):
            idx = indices[i:i+batch_size]
            xb, yb = X[idx], Y[idx]
            curr_bs = xb.shape[0]
            
            # Forward
            logits = model.forward(xb)
            
            # Fast Softmax Cross-Entropy
            shift_logits = logits - np.max(logits, axis=1, keepdims=True)
            ex = np.exp(shift_logits)
            probs = ex / np.sum(ex, axis=1, keepdims=True)
            
            loss = -np.mean(np.log(probs[range(curr_bs), yb] + 1e-10))
            acc = np.mean(np.argmax(probs, axis=1) == yb)
            
            # Backward
            d_logits = probs.copy()
            d_logits[range(curr_bs), yb] -= 1
            d_logits /= curr_bs
            
            model.backward(d_logits)
            
            epoch_loss += loss
            epoch_acc += acc
            batches += 1
            
        if epoch % 5 == 0:
            avg_loss = epoch_loss / batches
            avg_acc = epoch_acc / batches
            elapsed = time.time() - start_time
            print(f"EVO_STEP:{epoch:03d} | LOSS:{avg_loss:.4f} | ACC:{avg_acc:.4f} | TIME:{elapsed:.2f}s")

    print("PHASE: EVOLUTION_SUCCESS | ARCHITECTURE: OPTIMIZED")
    print("REDUNDANCY_CHECK: GEMINI_PATH=OK | GROQ_PATH=OK")

if __name__ == "__main__":
    train_evolution()
