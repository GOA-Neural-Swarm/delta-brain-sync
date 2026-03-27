import numpy as np
import time

class Parameter:
    def __init__(self, data, name=""):
        self.d = data.astype(np.float32)
        self.g = np.zeros_like(self.d)
        self.m = np.zeros_like(self.d)
        self.v = np.zeros_like(self.d)

class Module:
    def __init__(self):
        self.training = True
    def forward(self, x): raise NotImplementedError
    def backward(self, g): raise NotImplementedError
    def parameters(self):
        ps = []
        for v in self.__dict__.values():
            if isinstance(v, Parameter): ps.append(v)
            elif isinstance(v, Module): ps.extend(v.parameters())
            elif isinstance(v, list):
                for i in v:
                    if isinstance(i, Module): ps.extend(i.parameters())
        return ps

class Linear(Module):
    def __init__(self, i, o, bias=True):
        super().__init__()
        self.w = Parameter(np.random.randn(i, o) * np.sqrt(2.0 / i))
        self.b = Parameter(np.zeros((1, o))) if bias else None
    def forward(self, x):
        self.x = x
        out = x @ self.w.d
        if self.b: out += self.b.d
        return out
    def backward(self, g):
        self.w.g = self.x.T @ g
        if self.b: self.b.g = np.sum(g, axis=0, keepdims=True)
        return g @ self.w.d.T

class RMSNorm(Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.gamma = Parameter(np.ones((1, d)))
        self.eps = eps
    def forward(self, x):
        self.x = x
        self.rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        self.x_hat = x / self.rms
        return self.gamma.d * self.x_hat
    def backward(self, g):
        d = g.shape[-1]
        self.gamma.g = np.sum(g * self.x_hat, axis=0, keepdims=True)
        dx_hat = g * self.gamma.d
        return (dx_hat - self.x_hat * np.mean(dx_hat * self.x_hat, axis=-1, keepdims=True)) / self.rms

class SwiGLU(Module):
    def __init__(self, d, h):
        super().__init__()
        self.w1 = Linear(d, h, bias=False)
        self.w2 = Linear(d, h, bias=False)
        self.w3 = Linear(h, d, bias=False)
    def forward(self, x):
        self.x1 = self.w1.forward(x)
        self.x2 = self.w2.forward(x)
        self.sig = 1.0 / (1.0 + np.exp(-self.x1))
        self.swish = self.x1 * self.sig
        self.out = self.swish * self.x2
        return self.w3.forward(self.out)
    def backward(self, g):
        g = self.w3.backward(g)
        dx2 = g * self.swish
        dswish = g * self.x2
        dx1 = dswish * (self.sig * (1.0 + self.x1 * (1.0 - self.sig)))
        return self.w1.backward(dx1) + self.w2.backward(dx2)

class GELU(Module):
    def forward(self, x):
        self.x = x
        self.sig = 1.0 / (1.0 + np.exp(-1.702 * x))
        return x * self.sig
    def backward(self, g):
        s = self.sig
        return g * (s + 1.702 * self.x * s * (1.0 - s))

class SovereignBlock(Module):
    def __init__(self, d):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.gemini_path = SwiGLU(d, d * 4)
        self.norm2 = RMSNorm(d)
        self.groq_path = Linear(d, d * 4)
        self.groq_act = GELU()
        self.groq_proj = Linear(d * 4, d)
        self.gate = Parameter(np.zeros((1, d)))
    def forward(self, x):
        self.res = x
        nx = self.norm1.forward(x)
        self.out_a = self.gemini_path.forward(nx)
        self.out_b = self.groq_proj.forward(self.groq_act.forward(self.groq_path.forward(nx)))
        self.g_val = 1.0 / (1.0 + np.exp(-self.gate.d))
        return self.res + self.g_val * self.out_a + (1.0 - self.g_val) * self.out_b
    def backward(self, g_in):
        dg = self.g_val * (1.0 - self.g_val)
        self.gate.g = np.sum(g_in * (self.out_a - self.out_b) * dg, axis=0, keepdims=True)
        ga = self.gemini_path.backward(g_in * self.g_val)
        gb = self.groq_path.backward(self.groq_act.backward(self.groq_proj.backward(g_in * (1.0 - self.g_val))))
        return self.norm1.backward(ga + gb) + g_in

class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=0.01):
        self.params, self.lr, self.betas, self.eps, self.wd = params, lr, betas, eps, wd
        self.t = 0
    def step(self):
        self.t += 1
        a = self.lr * np.sqrt(1 - self.betas[1]**self.t) / (1 - self.betas[0]**self.t)
        for p in self.params:
            if self.wd > 0: p.d -= self.lr * self.wd * p.d
            p.m = self.betas[0] * p.m + (1 - self.betas[0]) * p.g
            p.v = self.betas[1] * p.v + (1 - self.betas[1]) * (p.g**2)
            p.d -= a * p.m / (np.sqrt(p.v) + self.eps)

class OMEGA_ASI(Module):
    def __init__(self, in_d, hid_d, out_d, depth=4):
        super().__init__()
        self.stem = Linear(in_d, hid_d)
        self.blocks = [SovereignBlock(hid_d) for _ in range(depth)]
        self.head_norm = RMSNorm(hid_d)
        self.head = Linear(hid_d, out_d)
        self.ps = self.parameters()
        self.opt = AdamW(self.ps, lr=1e-3, wd=0.05)
    def forward(self, x, training=True):
        self.training = training
        x = self.stem.forward(x)
        for b in self.blocks: x = b.forward(x)
        return self.head.forward(self.head_norm.forward(x))
    def backward(self, g):
        g = self.head_norm.backward(self.head.backward(g))
        for b in reversed(self.blocks): g = b.backward(g)
        self.stem.backward(g)
    def train_step(self, x, y):
        logits = self.forward(x, True)
        ex = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = ex / np.sum(ex, axis=1, keepdims=True)
        loss = -np.mean(np.sum(y * np.log(probs + 1e-12), axis=1))
        self.backward((probs - y) / y.shape[0])
        # Gradient Clipping
        gnorm = np.sqrt(sum(np.sum(p.g**2) for p in self.ps))
        if gnorm > 1.0:
            for p in self.ps: p.g /= gnorm
        self.opt.step()
        return loss

def get_data(n=5000, d=784, c=10):
    x = np.random.randn(n, d).astype(np.float32)
    w = np.random.randn(d, c).astype(np.float32)
    y_idx = np.argmax(x @ w + 0.05 * np.random.randn(n, c), axis=1)
    y = np.eye(c)[y_idx].astype(np.float32)
    return (x - np.mean(x)) / np.std(x), y

if __name__ == "__main__":
    X, Y = get_data(n=10000)
    model = OMEGA_ASI(784, 128, 10, depth=4)
    batch_size = 64
    epochs = 50
    
    print("OMEGA-ASI: ARCHITECTURAL EVOLUTION INITIALIZED")
    for e in range(1, epochs + 1):
        idx = np.random.permutation(len(X))
        losses = []
        t0 = time.time()
        for i in range(0, len(X), batch_size):
            bx, by = X[idx[i:i+batch_size]], Y[idx[i:i+batch_size]]
            losses.append(model.train_step(bx, by))
        
        dt = time.time() - t0
        val_logits = model.forward(X[:500], False)
        acc = np.mean(np.argmax(val_logits, axis=1) == np.argmax(Y[:500], axis=1))
        print(f"CYCLE {e:03} | LOSS: {np.mean(losses):.5f} | ACC: {acc:.4f} | TIME: {dt:.2f}s")
        if acc > 0.998: break
    print("EVOLUTIONARY STASIS REACHED.")
