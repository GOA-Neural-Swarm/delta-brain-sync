import numpy as np
import time

def stable_softmax(x, axis=-1):
    z = x - np.max(x, axis=axis, keepdims=True)
    n = np.exp(z)
    return n / (np.sum(n, axis=axis, keepdims=True) + 1e-12)

class Linear:
    def __init__(self, in_f, out_f, init_scale=1.0):
        std = np.sqrt(2.0 / in_f) * init_scale
        self.W = np.random.normal(0, std, (in_f, out_f)).astype(np.float32)
        self.b = np.zeros(out_f, dtype=np.float32)
        self.x, self.dW, self.db = None, None, None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return np.dot(dout, self.W.T)

    def params(self):
        return [{"ref": self.W, "grad": self.dW}, {"ref": self.b, "grad": self.db}]

class RMSNorm:
    def __init__(self, dim, eps=1e-6):
        self.g = np.ones(dim, dtype=np.float32)
        self.eps = eps
        self.x, self.rstd, self.dg = None, None, None

    def forward(self, x):
        self.x = x
        ms = np.mean(x**2, axis=-1, keepdims=True)
        self.rstd = 1.0 / np.sqrt(ms + self.eps)
        return self.g * (x * self.rstd)

    def backward(self, dout):
        nx = self.x * self.rstd
        self.dg = np.sum(dout * nx, axis=0)
        v = dout * self.g
        return self.rstd * (v - nx * np.mean(v * nx, axis=-1, keepdims=True))

    def params(self):
        return [{"ref": self.g, "grad": self.dg}]

class SwiGLU:
    def __init__(self, dim, h_dim):
        self.w12 = Linear(dim, h_dim * 2)
        self.w3 = Linear(h_dim, dim)
        self.h_dim = h_dim

    def forward(self, x):
        z = self.w12.forward(x)
        self.gate, self.act = z[:, :self.h_dim], z[:, self.h_dim:]
        self.sig = 1.0 / (1.0 + np.exp(-np.clip(self.gate, -10, 10)))
        self.swish = self.gate * self.sig
        return self.w3.forward(self.swish * self.act)

    def backward(self, dout):
        dc = self.w3.backward(dout)
        ds = dc * self.act
        da = dc * self.swish
        dg = ds * (self.sig * (1.0 + self.gate * (1.0 - self.sig)))
        return self.w12.backward(np.hstack([dg, da]))

    def params(self):
        return self.w12.params() + self.w3.params()

class MultiHeadAttention:
    def __init__(self, dim, heads=8):
        self.dim, self.heads = dim, heads
        self.hd = dim // heads
        self.scale = 1.0 / np.sqrt(self.hd)
        self.wqkv = Linear(dim, dim * 3)
        self.wo = Linear(dim, dim)

    def forward(self, x):
        b, d = x.shape
        qkv = self.wqkv.forward(x).reshape(b, 3, self.heads, self.hd)
        self.q, self.k, self.v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        dots = np.einsum('bhd,bmd->bhm', self.q, self.k) * self.scale
        self.att = stable_softmax(dots)
        out = np.einsum('bhm,bmd->bhd', self.att, self.v)
        return self.wo.forward(out.reshape(b, d))

    def backward(self, dout):
        b, d = dout.shape
        dctx = self.wo.backward(dout).reshape(b, self.heads, self.hd)
        datt = np.einsum('bhd,bmd->bhm', dctx, self.v)
        dv = np.einsum('bhm,bhd->bmd', self.att, dctx)
        ds = self.att * (datt - np.sum(self.att * datt, axis=-1, keepdims=True)) * self.scale
        dq = np.einsum('bhm,bmd->bhd', ds, self.k)
        dk = np.einsum('bhm,bhd->bmd', ds, self.q)
        return self.wqkv.backward(np.stack([dq, dk, dv], axis=1).reshape(b, 3 * d))

    def params(self):
        return self.wqkv.params() + self.wo.params()

class RedundantMoE:
    def __init__(self, dim):
        self.gemini = SwiGLU(dim, dim * 4) # High Capacity
        self.groq = Linear(dim, dim)      # High Speed
        self.gate = Linear(dim, 2)

    def forward(self, x):
        self.g_logits = self.gate.forward(x)
        self.p = stable_softmax(self.g_logits)
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
        self.ln1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim)
        self.ln2 = RMSNorm(dim)
        self.moe = RedundantMoE(dim)

    def forward(self, x):
        self.x1 = x
        self.norm1 = self.ln1.forward(x)
        self.x2 = self.x1 + self.attn.forward(self.norm1)
        self.norm2 = self.ln2.forward(self.x2)
        self.x3 = self.x2 + self.moe.forward(self.norm2)
        return self.x3

    def backward(self, dout):
        d_moe = self.moe.backward(dout)
        d_norm2 = self.ln2.backward(d_moe)
        dx2 = dout + d_norm2
        d_attn = self.attn.backward(self.ln1.forward(dx2)) # Approx norm for speed
        d_norm1 = self.ln1.backward(d_attn)
        return dx2 + d_norm1

    def params(self):
        return self.ln1.params() + self.attn.params() + self.ln2.params() + self.moe.params()

class SovereignArchitect:
    def __init__(self, in_d, h_d, out_d, depth=3):
        self.stem = Linear(in_d, h_d)
        self.blocks = [SovereignBlock(h_d) for _ in range(depth)]
        self.norm = RMSNorm(h_d)
        self.head = Linear(h_d, out_d, init_scale=0.1)

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
        return p + self.norm.params() + self.head.params()

class Lion:
    def __init__(self, params, lr=1e-4, b1=0.9, b2=0.99, wd=0.01):
        self.params, self.lr, self.b1, self.b2, self.wd = params, lr, b1, b2, wd
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
    return (X - np.mean(X)) / (np.std(X) + 1e-6), y

def train():
    N, D, K = 10000, 784, 10
    X, y = get_data(N, D, K)
    model = SovereignArchitect(D, 128, K, depth=2)
    opt = Lion(model.params(), lr=1e-4, wd=0.01)
    bs, epochs = 64, 20
    
    print(f"OMEGA-ASI | SOVEREIGN-V2 | PARAMS: {sum(p['ref'].size for p in model.params())}")
    
    for ep in range(epochs):
        idx = np.random.permutation(N)
        l_acc, a_acc, t0 = 0, 0, time.time()
        sched = 0.5 * (1 + np.cos(np.pi * ep / epochs))
        
        for i in range(0, N, bs):
            bi = idx[i : i + bs]
            xb, yb = X[bi], y[bi]
            m = xb.shape[0]
            
            probs = stable_softmax(model.forward(xb))
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
            
        print(f"EP:{ep:02d} | LOSS:{l_acc:.4f} | ACC:{a_acc:.4f} | SPEED:{N/(time.time()-t0):.0f}s/s")

if __name__ == "__main__":
    train()
