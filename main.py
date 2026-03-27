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

class LayerNorm(Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.gamma = Parameter(np.ones((1, d)))
        self.beta = Parameter(np.zeros((1, d)))
        self.eps = eps
    def forward(self, x):
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.std = np.sqrt(self.var + self.eps)
        self.x_hat = (x - self.mean) / self.std
        return self.gamma.d * self.x_hat + self.beta.d
    def backward(self, g):
        n, d = g.shape
        self.beta.g = np.sum(g, axis=0, keepdims=True)
        self.gamma.g = np.sum(g * self.x_hat, axis=0, keepdims=True)
        dx_hat = g * self.gamma.d
        return (d * dx_hat - np.sum(dx_hat, axis=-1, keepdims=True) - self.x_hat * np.sum(dx_hat * self.x_hat, axis=-1, keepdims=True)) / (d * self.std)

class GELU(Module):
    def forward(self, x):
        self.x = x
        self.sig = 1.0 / (1.0 + np.exp(-1.702 * x))
        return x * self.sig
    def backward(self, g):
        s = self.sig
        return g * (s + 1.702 * self.x * s * (1.0 - s))

class SiLU(Module):
    def forward(self, x):
        self.x = x
        self.sig = 1.0 / (1.0 + np.exp(-x))
        return x * self.sig
    def backward(self, g):
        s = self.sig
        return g * (s * (1.0 + self.x * (1.0 - s)))

class SovereignBlock(Module):
    def __init__(self, d):
        super().__init__()
        self.ln = LayerNorm(d)
        self.gemini_p1 = Linear(d, d * 4)
        self.gemini_act = GELU()
        self.gemini_p2 = Linear(d * 4, d)
        self.groq_p1 = Linear(d, d * 4)
        self.groq_act = SiLU()
        self.groq_p2 = Linear(d * 4, d)
        self.gate = Parameter(np.zeros((1, d)))
    def forward(self, x):
        self.res = x
        x = self.ln.forward(x)
        self.out_a = self.gemini_p2.forward(self.gemini_act.forward(self.gemini_p1.forward(x)))
        self.out_b = self.groq_p2.forward(self.groq_act.forward(self.groq_p1.forward(x)))
        self.g_val = 1.0 / (1.0 + np.exp(-self.gate.d))
        return self.res + self.g_val * self.out_a + (1.0 - self.g_val) * self.out_b
    def backward(self, g_in):
        dg = self.g_val * (1.0 - self.g_val)
        self.gate.g = np.sum(g_in * (self.out_a - self.out_b) * dg, axis=0, keepdims=True)
        ga = self.gemini_p1.backward(self.gemini_act.backward(self.gemini_p2.backward(g_in * self.g_val)))
        gb = self.groq_p1.backward(self.groq_act.backward(self.groq_p2.backward(g_in * (1.0 - self.g_val))))
        return self.ln.backward(ga + gb) + g_in

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
        self.head_ln = LayerNorm(hid_d)
        self.head = Linear(hid_d, out_d)
        self.ps = self.parameters()
        self.opt = AdamW(self.ps, lr=2e-3, wd=0.01)
    def forward(self, x, training=True):
        self.training = training
        x = self.stem.forward(x)
        for b in self.blocks: x = b.forward(x)
        return self.head.forward(self.head_ln.forward(x))
    def backward(self, g):
        g = self.head_ln.backward(self.head.backward(g))
        for b in reversed(self.blocks): g = b.backward(g)
        self.stem.backward(g)
    def train_step(self, x, y):
        logits = self.forward(x, True)
        ex = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = ex / np.sum(ex, axis=1, keepdims=True)
        loss = -np.mean(np.sum(y * np.log(probs + 1e-12), axis=1))
        self.backward((probs - y) / y.shape[0])
        self.opt.step()
        return loss

def get_data(n=10000, d=784, c=10):
    x = np.random.randn(n, d).astype(np.float32)
    w = np.random.randn(d, c).astype(np.float32)
    y_idx = np.argmax(x @ w + 0.1 * np.random.randn(n, c), axis=1)
    y = np.eye(c)[y_idx].astype(np.float32)
    return (x - np.mean(x)) / np.std(x), y

if __name__ == "__main__":
    X, Y = get_data()
    model = OMEGA_ASI(784, 256, 10, depth=3)
    batch_size = 128
    epochs = 100
    
    print("SOVEREIGN ARCHITECT: EVOLUTIONARY SEQUENCE START")
    for e in range(1, epochs + 1):
        idx = np.random.permutation(len(X))
        losses = []
        t0 = time.time()
        for i in range(0, len(X), batch_size):
            bx, by = X[idx[i:i+batch_size]], Y[idx[i:i+batch_size]]
            losses.append(model.train_step(bx, by))
        
        dt = time.time() - t0
        val_logits = model.forward(X[:1000], False)
        acc = np.mean(np.argmax(val_logits, axis=1) == np.argmax(Y[:1000], axis=1))
        print(f"CYCLE {e:03} | LOSS: {np.mean(losses):.5f} | ACC: {acc:.4f} | {dt:.2f}s")
        if acc > 0.995: break
    print("EVOLUTIONARY TARGET ATTAINED.")
