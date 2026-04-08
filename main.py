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

    def step(self, params, grads, clip_norm=1.0):
        self.t += 1
        total_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
        clip_coef = clip_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            grads = [g * clip_coef for g in grads]

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
    def __init__(self, p=0.1):
        self.p = p
        self.mask = None
    def forward(self, x, train=True):
        if not train: return x
        self.mask = (np.random.rand(*x.shape) > self.p) / (1.0 - self.p)
        return x * self.mask
    def backward(self, dout):
        return dout * self.mask

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

class SovereignRedundancy:
    def __init__(self, dim):
        self.path_gemini = Linear(dim, dim)
        self.path_groq = Linear(dim, dim)
        self.gate = np.ones((1, dim), dtype=np.float32) * 0.5

    def forward(self, x):
        self.x = x
        self.out_gemini = self.path_gemini.forward(x)
        self.out_groq = self.path_groq.forward(x)
        return self.gate * self.out_gemini + (1 - self.gate) * self.out_groq

    def backward(self, dout):
        d_gemini = dout * self.gate
        d_groq = dout * (1 - self.gate)
        self.dgate = np.sum(dout * (self.out_gemini - self.out_groq), axis=0, keepdims=True)
        dx_gemini = self.path_gemini.backward(d_gemini)
        dx_groq = self.path_groq.backward(d_groq)
        return dx_gemini + dx_groq

    def get_layers(self): return [self.path_gemini, self.path_groq]
    def get_params(self): return [self.gate]
    def get_grads(self): return [self.dgate]

class ResidualBlock:
    def __init__(self, dim, dropout_p=0.1):
        self.ln = LayerNorm(dim)
        self.l1 = Linear(dim, dim * 4)
        self.act = Swish()
        self.l2 = Linear(dim * 4, dim)
        self.drop = Dropout(dropout_p)
        self.redundancy = SovereignRedundancy(dim)

    def forward(self, x, train=True):
        self.res = x
        h = self.ln.forward(x)
        h = self.l1.forward(h)
        h = self.act.forward(h)
        h = self.l2.forward(h)
        h = self.drop.forward(h, train)
        h = self.redundancy.forward(h)
        return h + x

    def backward(self, dout):
        dh = self.redundancy.backward(dout)
        dh = self.drop.backward(dh)
        dh = self.l2.backward(dh)
        dh = self.act.backward(dh)
        dh = self.l1.backward(dh)
        dh = self.ln.backward(dh)
        return dh + dout

    def get_layers(self): 
        return [self.ln, self.l1, self.l2, self.redundancy]

class SovereignEngine:
    def __init__(self, in_d=784, h_d=256, out_d=10):
        self.layers = [
            Linear(in_d, h_d),
            ResidualBlock(h_d),
            ResidualBlock(h_d),
            LayerNorm(h_d),
            Linear(h_d, out_d)
        ]
        self.flat_layers = []
        for l in self.layers:
            if hasattr(l, 'get_layers'): 
                for sub in l.get_layers():
                    self.flat_layers.append(sub)
            else:
                self.flat_layers.append(l)
        
        params = []
        for l in self.flat_layers: params.extend(l.get_params())
        if hasattr(self.layers[1], 'redundancy'):
            params.extend(self.layers[1].redundancy.get_params())
            params.extend(self.layers[2].redundancy.get_params())
            
        self.params = params
        self.optimizer = AdamW(self.params, lr=1e-3, wd=0.05)

    def forward(self, x, train=True):
        for l in self.layers:
            if isinstance(l, (ResidualBlock, Dropout)):
                x = l.forward(x, train)
            else:
                x = l.forward(x)
        return x

    def backward(self, dout):
        for l in reversed(self.layers):
            dout = l.backward(dout)
        grads = []
        for l in self.flat_layers: grads.extend(l.get_grads())
        if hasattr(self.layers[1], 'redundancy'):
            grads.extend(self.layers[1].redundancy.get_grads())
            grads.extend(self.layers[2].redundancy.get_grads())
        self.optimizer.step(self.params, grads)

def train_evolution():
    np.random.seed(42)
    N, D, C = 1000, 784, 10
    X = np.random.randn(N, D).astype(np.float32)
    Y = np.random.randint(0, C, N)
    
    batch_size = 64
    model = SovereignEngine(D, 256, C)
    
    print("PHASE: RECURSIVE_EVOLUTION_INITIATED")
    start_time = time.time()
    
    for epoch in range(50):
        indices = np.random.permutation(N)
        epoch_loss = 0
        epoch_acc = 0
        
        for i in range(0, N, batch_size):
            idx = indices[i:i+batch_size]
            xb, yb = X[idx], Y[idx]
            
            logits = model.forward(xb, train=True)
            
            shift_logits = logits - np.max(logits, axis=1, keepdims=True)
            ex = np.exp(shift_logits)
            probs = ex / np.sum(ex, axis=1, keepdims=True)
            
            loss = -np.mean(np.log(probs[range(len(yb)), yb] + 1e-10))
            epoch_loss += loss * len(yb)
            epoch_acc += np.sum(np.argmax(probs, axis=1) == yb)
            
            d_logits = probs.copy()
            d_logits[range(len(yb)), yb] -= 1
            d_logits /= len(yb)
            
            model.backward(d_logits)
            
        if epoch % 5 == 0:
            avg_loss = epoch_loss / N
            avg_acc = epoch_acc / N
            print(f"EPOCH:{epoch:03d} | LOSS:{avg_loss:.4f} | ACC:{avg_acc:.4f} | TIME:{time.time()-start_time:.2f}s")

    print("PHASE: EVOLUTION_SUCCESS | STATUS: OMEGA_OPTIMIZED")

if __name__ == "__main__":
    train_evolution()
