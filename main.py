import numpy as np
import time

def softmax(x, axis=-1):
    max_x = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - max_x)
    return e_x / (np.sum(e_x, axis=axis, keepdims=True) + 1e-12)

class Linear:
    def __init__(self, in_f, out_f, scale=1.0):
        self.W = np.random.randn(in_f, out_f).astype(np.float32) * (np.sqrt(2.0 / in_f) * scale)
        self.b = np.zeros(out_f, dtype=np.float32)
        self.x, self.dW, self.db = None, None, None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0)
        return dout @ self.W.T

    def params(self):
        return [{"ref": self.W, "grad": self.dW}, {"ref": self.b, "grad": self.db}]

class RMSNorm:
    def __init__(self, dim, eps=1e-6):
        self.g = np.ones(dim, dtype=np.float32)
        self.eps = eps
        self.x, self.rstd, self.dg = None, None, None

    def forward(self, x):
        self.x = x
        var = np.mean(x**2, axis=-1, keepdims=True)
        self.rstd = 1.0 / np.sqrt(var + self.eps)
        return self.g * (x * self.rstd)

    def backward(self, dout):
        xr = self.x * self.rstd
        self.dg = np.sum(dout * xr, axis=0)
        dxr = dout * self.g
        return self.rstd * (dxr - xr * np.mean(dxr * xr, axis=-1, keepdims=True))

    def params(self):
        return [{"ref": self.g, "grad": self.dg}]

class SwiGLU:
    def __init__(self, dim, h_dim):
        self.w1 = Linear(dim, h_dim)
        self.w2 = Linear(dim, h_dim)
        self.w3 = Linear(h_dim, dim)
        self.z1, self.z2, self.sig = None, None, None

    def forward(self, x):
        self.z1 = self.w1.forward(x)
        self.z2 = self.w2.forward(x)
        self.sig = 1.0 / (1.0 + np.exp(-np.clip(self.z1, -12, 12)))
        return self.w3.forward((self.z1 * self.sig) * self.z2)

    def backward(self, dout):
        dz3 = self.w3.backward(dout)
        swish = self.z1 * self.sig
        dz2 = dz3 * swish
        dswish = dz3 * self.z2
        dz1 = dswish * (self.sig * (1.0 + self.z1 * (1.0 - self.sig)))
        return self.w1.backward(dz1) + self.w2.backward(dz2)

    def params(self):
        return self.w1.params() + self.w2.params() + self.w3.params()

class MultiHeadAttention:
    def __init__(self, dim, heads=8):
        self.dim, self.heads = dim, heads
        self.hd = dim // heads
        self.wq = Linear(dim, dim)
        self.wk = Linear(dim, dim)
        self.wv = Linear(dim, dim)
        self.wo = Linear(dim, dim)
        self.q, self.k, self.v, self.att, self.ctx = [None]*5

    def forward(self, x):
        b, d = x.shape
        self.q = self.wq.forward(x).reshape(b, self.heads, self.hd)
        self.k = self.wk.forward(x).reshape(b, self.heads, self.hd)
        self.v = self.wv.forward(x).reshape(b, self.heads, self.hd)
        
        s = np.einsum('bhd,bmd->bhm', self.q, self.k) / np.sqrt(self.hd)
        self.att = softmax(s, axis=-1)
        self.ctx = np.einsum('bhm,bmd->bhd', self.att, self.v).reshape(b, d)
        return self.wo.forward(self.ctx)

    def backward(self, dout):
        dctx = self.wo.backward(dout)
        b, d = dctx.shape
        dctx_r = dctx.reshape(b, self.heads, self.hd)
        
        datt = np.einsum('bhd,bmd->bhm', dctx_r, self.v)
        dv = np.einsum('bhm,bhd->bmd', self.att, dctx_r)
        
        ds = self.att * (datt - np.sum(self.att * datt, axis=-1, keepdims=True))
        ds /= np.sqrt(self.hd)
        
        dq = np.einsum('bhm,bmd->bhd', ds, self.k)
        dk = np.einsum('bhm,bhd->bmd', ds, self.q)
        
        return self.wq.backward(dq.reshape(b, d)) + self.wk.backward(dk.reshape(b, d)) + self.wv.backward(dv.reshape(b, d))

    def params(self):
        return self.wq.params() + self.wk.params() + self.wv.params() + self.wo.params()

class RedundantMoE:
    def __init__(self, dim):
        self.gemini = SwiGLU(dim, dim * 4) # High Capacity
        self.groq = SwiGLU(dim, dim * 2)   # High Throughput
        self.gate = Linear(dim, 2)
        self.p, self.o1, self.o2 = None, None, None

    def forward(self, x):
        g = self.gate.forward(x)
        self.p = softmax(g)
        self.o1 = self.gemini.forward(x)
        self.o2 = self.groq.forward(x)
        return self.p[:, 0:1] * self.o1 + self.p[:, 1:2] * self.o2

    def backward(self, dout):
        p1, p2 = self.p[:, 0:1], self.p[:, 1:2]
        dx1 = self.gemini.backward(dout * p1)
        dx2 = self.groq.backward(dout * p2)
        dp1 = np.sum(dout * self.o1, axis=-1, keepdims=True)
        dp2 = np.sum(dout * self.o2, axis=-1, keepdims=True)
        dg = self.p * (np.hstack([dp1, dp2]) - np.sum(self.p * np.hstack([dp1, dp2]), axis=-1, keepdims=True))
        return dx1 + dx2 + self.gate.backward(dg)

    def params(self):
        return self.gemini.params() + self.groq.params() + self.gate.params()

class SovereignBlock:
    def __init__(self, dim):
        self.n1 = RMSNorm(dim)
        self.at = MultiHeadAttention(dim)
        self.n2 = RMSNorm(dim)
        self.moe = RedundantMoE(dim)

    def forward(self, x):
        x = x + self.at.forward(self.n1.forward(x))
        x = x + self.moe.forward(self.n2.forward(x))
        return x

    def backward(self, dout):
        dm = self.moe.backward(dout)
        dn2 = self.n2.backward(dm)
        dx_m = dout + dn2
        da = self.at.backward(dx_m)
        dn1 = self.n1.backward(da)
        return dx_m + dn1

    def params(self):
        return self.n1.params() + self.at.params() + self.n2.params() + self.moe.params()

class SovereignArchitect:
    def __init__(self, in_d, h_d, out_d, depth=4):
        self.stem = Linear(in_d, h_d)
        self.blocks = [SovereignBlock(h_d) for _ in range(depth)]
        self.norm = RMSNorm(h_d)
        self.head = Linear(h_d, out_d, scale=0.1)

    def forward(self, x):
        x = self.stem.forward(x)
        for b in self.blocks: x = b.forward(x)
        return self.head.forward(self.norm.forward(x))

    def backward(self, dout):
        dout = self.norm.backward(self.head.backward(dout))
        for b in reversed(self.blocks): dout = b.backward(dout)
        self.stem.backward(dout)

    def params(self):
        p = self.stem.params()
        for b in self.blocks: p.extend(b.params())
        p.extend(self.norm.params() + self.head.params())
        return p

class Lion:
    def __init__(self, params, lr=1e-4, b1=0.9, b2=0.99, wd=0.01):
        self.params = params
        self.lr, self.b1, self.b2, self.wd = lr, b1, b2, wd
        self.m = [np.zeros_like(p["ref"]) for p in params]

    def step(self, scale=1.0):
        lr = self.lr * scale
        for i, p in enumerate(self.params):
            if p["grad"] is None: continue
            w, g = p["ref"], p["grad"]
            if self.wd > 0: w -= lr * self.wd * w
            u = np.sign(self.b1 * self.m[i] + (1.0 - self.b1) * g)
            w -= lr * u
            self.m[i] = self.b2 * self.m[i] + (1.0 - self.b2) * g

def get_data(n, d, k):
    X = np.random.randn(n, d).astype(np.float32)
    y = np.random.randint(0, k, n)
    c = np.random.randn(k, d).astype(np.float32) * 4.0
    X += c[y]
    X = (X - X.mean()) / (X.std() + 1e-6)
    return X, y

def train():
    N, D, K = 20000, 784, 10
    X, y = get_data(N, D, K)
    model = SovereignArchitect(D, 256, K, depth=4)
    opt = Lion(model.params(), lr=1e-4, wd=0.01)
    
    bs, epochs = 128, 50
    print(f"OMEGA-ASI | ARCHITECTURE: SOVEREIGN-V14 | PARAMS: {sum(p['ref'].size for p in model.params())}")

    for ep in range(epochs):
        idx = np.random.permutation(N)
        l_acc, a_acc, t0 = 0, 0, time.time()
        sched = 0.5 * (1 + np.cos(np.pi * ep / epochs))

        for i in range(0, N, bs):
            bi = idx[i : i + bs]
            xb, yb = X[bi], y[bi]
            m = xb.shape[0]

            logits = model.forward(xb)
            probs = softmax(logits)
            
            loss = -np.mean(np.log(probs[range(m), yb] + 1e-10))
            l_acc += loss * (m / N)
            a_acc += np.mean(np.argmax(probs, axis=1) == yb) * (m / N)

            dout = probs.copy()
            dout[range(m), yb] -= 1
            model.backward(dout / m)

            gn = np.sqrt(sum(np.sum(p["grad"]**2) for p in model.params() if p["grad"] is not None))
            if gn > 1.0:
                for p in model.params():
                    if p["grad"] is not None: p["grad"] /= (gn + 1e-6)

            opt.step(scale=sched)

        dt = time.time() - t0
        print(f"EP:{ep:02d} | LOSS:{l_acc:.4f} | ACC:{a_acc:.4f} | SPEED:{N/dt:.0f} samples/s | LR:{opt.lr*sched:.7f}")

if __name__ == "__main__":
    train()
