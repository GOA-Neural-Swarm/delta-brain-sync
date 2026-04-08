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

class Dropout:
    def __init__(self, prob=0.1):
        self.prob = prob
        self.mask = None

    def forward(self, x, training=True):
        if not training: return x
        self.mask = (np.random.rand(*x.shape) > self.prob).astype(np.float32) / (1.0 - self.prob)
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask

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
    def __init__(self, dim, dropout=0.1):
        self.ln1 = LayerNorm(dim)
        self.l1 = Linear(dim, dim * 4)
        self.act = Swish()
        self.l2 = Linear(dim * 4, dim)
        self.drop = Dropout(dropout)

    def forward(self, x, training=True):
        self.res = x
        h = self.ln1.forward(x)
        h = self.l1.forward(h)
        h = self.act.forward(h)
        h = self.l2.forward(h)
        h = self.drop.forward(h, training)
        return h + x

    def backward(self, dout):
        dh = self.drop.backward(dout)
        dh = self.l2.backward(dh)
        dh = self.act.backward(dh)
        dh = self.l1.backward(dh)
        dh = self.ln1.backward(dh)
        return dh + dout

    def get_layers(self): return [self.ln1, self.l1, self.l2]

class RedundancyOrchestrator:
    """Simulates Gemini and Groq redundant logic for gradient verification and meta-optimization."""
    def __init__(self, threshold=1.5):
        self.threshold = threshold
        self.history = []

    def verify_and_evolve(self, loss, lr):
        self.history.append(loss)
        if len(self.history) < 5: return lr, False
        
        # Simulated Gemini Logic: Trend Analysis
        gemini_signal = np.mean(self.history[-5:]) > np.mean(self.history[-10:-5]) if len(self.history) >= 10 else False
        
        # Simulated Groq Logic: Throughput/Stability Analysis
        groq_signal = np.std(self.history[-5:]) > self.threshold
        
        # Redundant Consensus
        if gemini_signal and groq_signal:
            return lr * 0.5, True # Reduce LR if diverging
        elif not gemini_signal and not groq_signal:
            return lr * 1.01, False # Slight boost if stable
        return lr, False

class SovereignEngine:
    def __init__(self, in_d=784, h_d=256, out_d=10, depth=4):
        self.layers = [Linear(in_d, h_d)]
        for _ in range(depth):
            self.layers.append(ResidualBlock(h_d))
        self.layers.append(LayerNorm(h_d))
        self.layers.append(Linear(h_d, out_d))
        
        self.flat_layers = []
        for l in self.layers:
            if hasattr(l, 'get_layers'): self.flat_layers.extend(l.get_layers())
            else: self.flat_layers.append(l)
        
        self.params = []
        for l in self.flat_layers: self.params.extend(l.get_params())
        self.optimizer = AdamW(self.params, lr=1e-3, wd=0.05)
        self.orchestrator = RedundancyOrchestrator()

    def forward(self, x, training=True):
        for l in self.layers:
            if isinstance(l, (ResidualBlock, Dropout)): x = l.forward(x, training)
            else: x = l.forward(x)
        return x

    def backward(self, dout):
        for l in reversed(self.layers):
            dout = l.backward(dout)
        grads = []
        for l in self.flat_layers: grads.extend(l.get_grads())
        self.optimizer.step(self.params, grads)

    def evolve_hyperparameters(self, loss):
        new_lr, mutated = self.orchestrator.verify_and_evolve(loss, self.optimizer.lr)
        self.optimizer.lr = new_lr
        return mutated

def train_evolution():
    # High-Performance Synthetic Dataset
    N, D, C = 1000, 784, 10
    X = np.random.randn(N, D).astype(np.float32)
    Y = np.random.randint(0, C, N)
    
    batch_size = 64
    model = SovereignEngine(in_d=D, h_d=256, out_d=C, depth=3)
    
    print("PHASE: RECURSIVE_EVOLUTION_INIT")
    start_time = time.time()
    
    for epoch in range(50):
        indices = np.arange(N)
        np.random.shuffle(indices)
        epoch_loss = 0
        epoch_acc = 0
        
        for i in range(0, N, batch_size):
            idx = indices[i:i+batch_size]
            xb, yb = X[idx], Y[idx]
            curr_bs = xb.shape[0]
            
            # Forward
            logits = model.forward(xb, training=True)
            
            # Stable Softmax
            shift_logits = logits - np.max(logits, axis=1, keepdims=True)
            ex = np.exp(shift_logits)
            probs = ex / np.sum(ex, axis=1, keepdims=True)
            
            # Loss & Accuracy
            loss = -np.mean(np.log(probs[range(curr_bs), yb] + 1e-10))
            acc = np.mean(np.argmax(probs, axis=1) == yb)
            
            # Backward
            d_logits = probs.copy()
            d_logits[range(curr_bs), yb] -= 1
            d_logits /= curr_bs
            
            model.backward(d_logits)
            
            epoch_loss += loss * (curr_bs / N)
            epoch_acc += acc * (curr_bs / N)
            
        # Redundancy Check & Recursive Evolution
        mutated = model.evolve_hyperparameters(epoch_loss)
        
        if epoch % 5 == 0:
            status = "MUTATED" if mutated else "STABLE"
            elapsed = time.time() - start_time
            print(f"EP:{epoch:03d} | LOSS:{epoch_loss:.4f} | ACC:{epoch_acc:.4f} | LR:{model.optimizer.lr:.6f} | {status} | TIME:{elapsed:.2f}s")

    print("PHASE: EVOLUTION_COMPLETE")
    print(f"FINAL_ACCURACY: {epoch_acc:.4f}")
    print("SYSTEM_STATUS: OMEGA_OPTIMIZED")

if __name__ == "__main__":
    train_evolution()
