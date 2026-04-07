import numpy as np

class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=0.01):
        self.lr, self.beta1, self.beta2, self.eps, self.wd = lr, betas[0], betas[1], eps, wd
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)
        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i]**2)
            params[i] -= lr_t * (self.m[i] / (np.sqrt(self.v[i]) + self.eps) + self.wd * params[i])

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
        self.dgamma = np.sum(dout * self.x_hat, axis=0, keepdims=True)
        self.dbeta = np.sum(dout, axis=0, keepdims=True)
        dx = (1. / D) * self.std_inv * (D * dx_hat - np.sum(dx_hat, axis=-1, keepdims=True) - self.x_hat * np.sum(dx_hat * self.x_hat, axis=-1, keepdims=True))
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

class EvolutionBlock:
    """Integrated High-Performance Module (Consolidating Gemini and Groq logic)"""
    def __init__(self, dim):
        self.ln = LayerNorm(dim)
        self.l1 = Linear(dim, dim * 4)
        self.act = Swish()
        self.l2 = Linear(dim * 4, dim)

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

    def get_sub_layers(self): return [self.ln, self.l1, self.l2]

class SovereignEngine:
    def __init__(self, in_d=784, h_d=256, out_d=10, depth=3):
        self.layers = [Linear(in_d, h_d)]
        for _ in range(depth):
            self.layers.append(EvolutionBlock(h_d))
        self.layers.append(LayerNorm(h_d))
        self.layers.append(Linear(h_d, out_d))
        
        self.flat_layers = []
        for l in self.layers:
            if hasattr(l, 'get_sub_layers'): self.flat_layers.extend(l.get_sub_layers())
            else: self.flat_layers.append(l)

        self.params = []
        for l in self.flat_layers: self.params.extend(l.get_params())
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
    np.random.seed(42)
    N, D, H, C = 1000, 784, 256, 10
    X = np.random.randn(N, D).astype(np.float32)
    Y = np.random.randint(0, C, N)
    
    model = SovereignEngine(in_d=D, h_d=H, out_d=C, depth=4)
    batch_size = 64
    epochs = 50

    print("PHASE: RECURSIVE_EVOLUTION_INITIATED")
    for epoch in range(epochs):
        indices = np.random.permutation(N)
        epoch_loss = 0
        epoch_acc = 0
        
        for i in range(0, N, batch_size):
            idx = indices[i:i+batch_size]
            xb, yb = X[idx], Y[idx]
            
            # Forward
            logits = model.forward(xb)
            
            # Softmax Cross-Entropy
            ex = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = ex / np.sum(ex, axis=1, keepdims=True)
            
            m = yb.shape[0]
            loss = -np.mean(np.log(probs[range(m), yb] + 1e-10))
            acc = np.mean(np.argmax(probs, axis=1) == yb)
            
            # Backward
            d_logits = probs.copy()
            d_logits[range(m), yb] -= 1
            d_logits /= m
            model.backward(d_logits)
            
            epoch_loss += loss * (m / N)
            epoch_acc += acc * (m / N)

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"EPOCH:{epoch:03d} | LOSS:{epoch_loss:.4f} | ACC:{epoch_acc:.4f}")

    print("PHASE: EVOLUTION_SUCCESS | STATUS: OPTIMIZED")

if __name__ == "__main__":
    train_evolution()
