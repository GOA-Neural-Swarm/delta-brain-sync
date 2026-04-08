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

    def step(self, params, grads, lr_scale=1.0):
        self.t += 1
        curr_lr = self.lr * lr_scale
        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i]**2)
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            params[i] -= curr_lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.wd * params[i])

class RMSNorm:
    def __init__(self, dim, eps=1e-6):
        self.g = np.ones((1, dim), dtype=np.float32)
        self.eps = eps

    def forward(self, x):
        self.x = x
        self.inv_rms = 1.0 / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        self.x_hat = x * self.inv_rms
        return self.g * self.x_hat

    def backward(self, dout):
        self.dg = np.sum(dout * self.x_hat, axis=0, keepdims=True)
        dx_hat = dout * self.g
        n = self.x.shape[-1]
        dx = (1.0 / n) * self.inv_rms * (n * dx_hat - self.x_hat * np.sum(dx_hat * self.x_hat, axis=-1, keepdims=True))
        return dx

    def get_params(self): return [self.g]
    def get_grads(self): return [self.dg]

class GeGLU:
    def forward(self, x):
        self.x = x
        self.tanh = np.tanh(0.79788456 * (x + 0.044715 * x**3))
        self.gelu = 0.5 * x * (1 + self.tanh)
        return x * self.gelu

    def backward(self, dout):
        # Simplified derivative for high-performance approximation
        pdf = 0.5 * (1 + self.tanh) + 0.5 * self.x * (1 - self.tanh**2) * 0.79788456 * (1 + 3 * 0.044715 * self.x**2)
        d_gelu = self.gelu + self.x * pdf
        return dout * d_gelu

class Linear:
    def __init__(self, in_d, out_d):
        scale = np.sqrt(2.0 / in_d)
        self.W = (np.random.randn(in_d, out_d) * scale).astype(np.float32)
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

class SovereignBlock:
    def __init__(self, dim):
        self.norm = RMSNorm(dim)
        self.alpha_w1 = Linear(dim, dim * 2)
        self.alpha_act = GeGLU()
        self.alpha_w2 = Linear(dim * 2, dim)
        self.beta_w1 = Linear(dim, dim * 2)
        self.beta_act = GeGLU()
        self.beta_w2 = Linear(dim * 2, dim)

    def forward(self, x):
        self.res = x
        h = self.norm.forward(x)
        self.a = self.alpha_w2.forward(self.alpha_act.forward(self.alpha_w1.forward(h)))
        self.b = self.beta_w2.forward(self.beta_act.forward(self.beta_w1.forward(h)))
        return self.res + 0.5 * (self.a + self.b)

    def backward(self, dout):
        d_consensus = 0.5 * dout
        da = self.alpha_w1.backward(self.alpha_act.backward(self.alpha_w2.backward(d_consensus)))
        db = self.beta_w1.backward(self.beta_act.backward(self.beta_w2.backward(d_consensus)))
        return self.norm.backward(da + db) + dout

    def get_layers(self):
        return [self.norm, self.alpha_w1, self.alpha_w2, self.beta_w1, self.beta_w2]

class OMEGA_ASI_Core:
    def __init__(self, in_d=784, h_d=256, out_d=10, depth=4):
        self.stem = Linear(in_d, h_d)
        self.blocks = [SovereignBlock(h_d) for _ in range(depth)]
        self.head_norm = RMSNorm(h_d)
        self.head = Linear(h_d, out_d)
        
        self.layers = [self.stem]
        for b in self.blocks: self.layers.extend(b.get_layers())
        self.layers.extend([self.head_norm, self.head])
        
        self.params = []
        for l in self.layers: self.params.extend(l.get_params())
        self.optimizer = AdamW(self.params, lr=2e-3, wd=0.02)

    def forward(self, x):
        x = self.stem.forward(x)
        for b in self.blocks: x = b.forward(x)
        return self.head.forward(self.head_norm.forward(x))

    def backward(self, dout, lr_scale=1.0):
        dout = self.head_norm.backward(self.head.backward(dout))
        for b in reversed(self.blocks): dout = b.backward(dout)
        self.stem.backward(dout)
        
        grads = []
        for l in self.layers: grads.extend(l.get_grads())
        self.optimizer.step(self.params, grads, lr_scale)

def train_evolution():
    np.random.seed(42)
    N, D, H, C = 2048, 784, 256, 10
    X = np.random.randn(N, D).astype(np.float32)
    Y = np.random.randint(0, C, N)
    
    model = OMEGA_ASI_Core(D, H, C, depth=4)
    batch_size = 64
    epochs = 50
    
    print("PHASE: OMEGA_EVOLUTION_START")
    start_time = time.time()
    
    for epoch in range(epochs):
        idx = np.random.permutation(N)
        X, Y = X[idx], Y[idx]
        
        epoch_loss, epoch_acc = 0, 0
        lr_scale = 0.5 * (1 + np.cos(np.pi * epoch / epochs))
        
        for i in range(0, N, batch_size):
            xb = X[i:i+batch_size]
            yb = Y[i:i+batch_size]
            m = xb.shape[0]
            
            logits = model.forward(xb)
            probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs /= np.sum(probs, axis=1, keepdims=True)
            
            loss = -np.mean(np.log(probs[range(m), yb] + 1e-10))
            acc = np.mean(np.argmax(probs, axis=1) == yb)
            
            epoch_loss += loss * (m / N)
            epoch_acc += acc * (m / N)
            
            d_logits = probs.copy()
            d_logits[range(m), yb] -= 1
            d_logits /= m
            
            model.backward(d_logits, lr_scale=lr_scale)
            
        if epoch % 5 == 0 or epoch == epochs - 1:
            dt = time.time() - start_time
            print(f"EVO:{epoch:03d} | LOSS:{epoch_loss:.4f} | ACC:{epoch_acc:.4f} | LRS:{lr_scale:.3f} | T:{dt:.2f}s")
            
    print("PHASE: EVOLUTION_COMPLETE | STATUS: SOVEREIGN")

if __name__ == "__main__":
    train_evolution()
