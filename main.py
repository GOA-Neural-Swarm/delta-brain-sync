
import numpy as np
import time

class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, wd=0.01):
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
            params[i] -= curr_lr * self.wd * params[i]
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i]**2)
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            params[i] -= curr_lr * m_hat / (np.sqrt(v_hat) + self.eps)

class RMSNorm:
    def __init__(self, dim, eps=1e-6):
        self.scale = np.ones((1, dim), dtype=np.float32)
        self.eps = eps

    def forward(self, x):
        self.x = x
        self.norm = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        self.x_normed = x / self.norm
        return self.scale * self.x_normed

    def backward(self, dout):
        dscale = np.sum(dout * self.x_normed, axis=0, keepdims=True)
        dx_normed = dout * self.scale
        dnorm = np.sum(dx_normed * self.x * -1.0 / (self.norm**2), axis=-1, keepdims=True)
        dx = (dx_normed / self.norm) + (dnorm * self.x / (self.x.shape[-1] * self.norm))
        self.dscale = dscale
        return dx

    def get_params(self): return [self.scale]
    def get_grads(self): return [self.dscale]

class GeGLU:
    def forward(self, x):
        self.x = x
        self.gate = 0.5 * x * (1 + np.tanh(0.79788456 * (x + 0.044715 * x**3)))
        return x * self.gate

    def backward(self, dout):
        tanh_out = np.tanh(0.79788456 * (self.x + 0.044715 * self.x**3))
        pdf = 0.79788456 * (1 + 3 * 0.044715 * self.x**2) * (1 - tanh_out**2)
        d_gate = 0.5 * (1 + tanh_out) + 0.5 * self.x * pdf
        return dout * (self.gate + self.x * d_gate)

class Linear:
    def __init__(self, in_d, out_d):
        limit = np.sqrt(6 / (in_d + out_d))
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

class Gemini:
    def __init__(self, dim):
        self.norm = RMSNorm(dim)
        self.w_alpha = Linear(dim, dim * 2)
        self.act_alpha = GeGLU()
        self.proj_alpha = Linear(dim * 2, dim)
        self.w_beta = Linear(dim, dim * 2)
        self.act_beta = GeGLU()
        self.proj_beta = Linear(dim * 2, dim)

    def forward(self, x):
        self.res = x
        h = self.norm.forward(x)
        self.out_alpha = self.proj_alpha.forward(self.act_alpha.forward(self.w_alpha.forward(h)))
        self.out_beta = self.proj_beta.forward(self.act_beta.forward(self.w_beta.forward(h)))
        return self.res + 0.5 * (self.out_alpha + self.out_beta)

    def backward(self, dout):
        d_consensus = 0.5 * dout
        db = self.proj_beta.backward(d_consensus)
        db = self.act_beta.backward(db)
        db = self.w_beta.backward(db)
        da = self.proj_alpha.backward(d_consensus)
        da = self.act_alpha.backward(da)
        da = self.w_alpha.backward(da)
        dn = self.norm.backward(da + db)
        return dn + dout

    def get_layers(self):
        return [self.norm, self.w_alpha, self.proj_alpha, self.w_beta, self.proj_beta]

class Groq:
    def __init__(self, dim):
        self.norm = RMSNorm(dim)
        self.w_alpha = Linear(dim, dim * 2)
        self.act_alpha = GeGLU()
        self.proj_alpha = Linear(dim * 2, dim)
        self.w_beta = Linear(dim, dim * 2)
        self.act_beta = GeGLU()
        self.proj_beta = Linear(dim * 2, dim)

    def forward(self, x):
        self.res = x
        h = self.norm.forward(x)
        self.out_alpha = self.proj_alpha.forward(self.act_alpha.forward(self.w_alpha.forward(h)))
        self.out_beta = self.proj_beta.forward(self.act_beta.forward(self.w_beta.forward(h)))
        return self.res + 0.5 * (self.out_alpha + self.out_beta)

    def backward(self, dout):
        d_consensus = 0.5 * dout
        db = self.proj_beta.backward(d_consensus)
        db = self.act_beta.backward(db)
        db = self.w_beta.backward(db)
        da = self.proj_alpha.backward(d_consensus)
        da = self.act_alpha.backward(da)
        da = self.w_alpha.backward(da)
        dn = self.norm.backward(da + db)
        return dn + dout

    def get_layers(self):
        return [self.norm, self.w_alpha, self.proj_alpha, self.w_beta, self.proj_beta]

class SovereignEngine:
    def __init__(self, in_d=784, h_d=256, out_d=10, depth=3):
        self.stem = Linear(in_d, h_d)
        self.blocks = [Gemini(h_d) if i % 2 == 0 else Groq(h_d) for i in range(depth)]
        self.head_norm = RMSNorm(h_d)
        self.head = Linear(h_d, out_d)
        self.flat_layers = [self.stem]
        for b in self.blocks: self.flat_layers.extend(b.get_layers())
        self.flat_layers.extend([self.head_norm, self.head])
        params = []
        for l in self.flat_layers: params.extend(l.get_params())
        self.params = params
        self.optimizer = AdamW(self.params, lr=1e-3, wd=0.05)

    def forward(self, x):
        x = self.stem.forward(x)
        for b in self.blocks: x = b.forward(x)
        x = self.head_norm.forward(x)
        return self.head.forward(x)

    def backward(self, dout, lr_scale=1.0):
        dout = self.head.backward(dout)
        dout = self.head_norm.backward(dout)
        for b in reversed(self.blocks): dout = b.backward(dout)
        self.stem.backward(dout)
        grads = []
        for l in self.flat_layers: grads.extend(l.get_grads())
        self.optimizer.step(self.params, grads, lr_scale)

def train_evolution():
    np.random.seed(42)
    N, D, H, C = 1024, 784, 256, 10
    X = np.random.randn(N, D).astype(np.float32)
    Y = np.random.randint(0, C, N)
    model = SovereignEngine(D, H, C, depth=4)
    batch_size = 128
    epochs = 100
    print("PHASE: HIGH_PERFORMANCE_EVOLUTION_INIT")
    start_time = time.time()
    for epoch in range(epochs):
        indices = np.random.permutation(N)
        epoch_loss = 0
        epoch_acc = 0
        lr_scale = 0.5 * (1 + np.cos(np.pi * epoch / epochs))
        for i in range(0, N, batch_size):
            batch_idx = indices[i:i+batch_size]
            xb, yb = X[batch_idx], Y[batch_idx]
            logits = model.forward(xb)
            shift_logits = logits - np.max(logits, axis=1, keepdims=True)
            ex = np.exp(shift_logits)
            probs = ex / np.sum(ex, axis=1, keepdims=True)
            m = yb.shape[0]
            loss = -np.mean(np.log(probs[range(m), yb] + 1e-10))
            acc = np.mean(np.argmax(probs, axis=1) == yb)
            epoch_loss += loss * (m / N)
            epoch_acc += acc * (m / N)
            d_logits = probs.copy()
            d_logits[range(m), yb] -= 1
            d_logits /= m
            model.backward(d_logits, lr_scale=lr_scale)
        if epoch % 10 == 0 or epoch == epochs - 1:
            elapsed = time.time() - start_time
            print(f"STEP:{epoch:03d} | LOSS:{epoch_loss:.4f} | ACC:{epoch_acc:.4f} | LR_S:{lr_scale:.3f} | TIME:{elapsed:.2f}s")
    print("PHASE: EVOLUTION_COMPLETE | STATUS: SUPREME")

if __name__ == "__main__":
    train_evolution()
