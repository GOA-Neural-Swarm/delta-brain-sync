import numpy as np
import time

def fast_softmax(x, axis=-1):
    c = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - c)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)

class Linear:
    def __init__(self, in_f, out_f, scale=1.0):
        self.W = np.random.randn(in_f, out_f).astype(np.float32) * (np.sqrt(2.0 / in_f) * scale)
        self.b = np.zeros(out_f, dtype=np.float32)
        self.x, self.dW, self.db = None, None, None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return np.dot(dout, self.W.T)

class RMSNorm:
    def __init__(self, dim, eps=1e-6):
        self.g = np.ones(dim, dtype=np.float32)
        self.eps = eps
        self.x, self.rstd, self.dg = None, None, None

    def forward(self, x):
        self.x = x
        self.rstd = 1.0 / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return self.g * (x * self.rstd)

    def backward(self, dout):
        nx = self.x * self.rstd
        self.dg = np.sum(dout * nx, axis=0)
        v = (dout * self.g)
        return self.rstd * (v - nx * np.mean(v * nx, axis=-1, keepdims=True))

class GeminiExpert:
    def __init__(self, dim, h_dim):
        self.w1 = Linear(dim, h_dim)
        self.w2 = Linear(dim, h_dim)
        self.w3 = Linear(h_dim, dim)

    def forward(self, x):
        self.z1 = self.w1.forward(x)
        self.z2 = self.w2.forward(x)
        self.sig = 1.0 / (1.0 + np.exp(-np.clip(self.z1, -10, 10)))
        self.swish = self.z1 * self.sig
        return self.w3.forward(self.swish * self.z2)

    def backward(self, dout):
        d3 = self.w3.backward(dout)
        dz2 = d3 * self.swish
        dswish = d3 * self.z2
        dz1 = dswish * (self.sig * (1.0 + self.z1 * (1.0 - self.sig)))
        return self.w1.backward(dz1) + self.w2.backward(dz2)

class GroqExpert:
    def __init__(self, dim, h_dim):
        self.w1 = Linear(dim, h_dim)
        self.w2 = Linear(h_dim, dim)

    def forward(self, x):
        self.z = self.w1.forward(x)
        self.mask = (self.z > 0).astype(np.float32)
        return self.w2.forward(self.z * self.mask)

    def backward(self, dout):
        dz = self.w2.backward(dout) * self.mask
        return self.w1.backward(dz)

class RedundantMoE:
    def __init__(self, dim):
        self.gemini = GeminiExpert(dim, dim * 4)
        self.groq = GroqExpert(dim, dim * 4)
        self.gate = Linear(dim, 2)

    def forward(self, x):
        self.probs = fast_softmax(self.gate.forward(x))
        self.o1 = self.gemini.forward(x)
        self.o2 = self.groq.forward(x)
        return self.probs[:, 0:1] * self.o1 + self.probs[:, 1:2] * self.o2

    def backward(self, dout):
        p1, p2 = self.probs[:, 0:1], self.probs[:, 1:2]
        dx = self.gemini.backward(dout * p1) + self.groq.backward(dout * p2)
        dp1 = np.sum(dout * self.o1, axis=-1, keepdims=True)
        dp2 = np.sum(dout * self.o2, axis=-1, keepdims=True)
        dp = np.concatenate([dp1, dp2], axis=-1)
        dg = self.probs * (dp - np.sum(self.probs * dp, axis=-1, keepdims=True))
        return dx + self.gate.backward(dg)

class MultiHeadAttention:
    def __init__(self, dim, heads=8):
        self.dim, self.heads = dim, heads
        self.hd = dim // heads
        self.scale = 1.0 / np.sqrt(self.hd)
        self.wq = Linear(dim, dim)
        self.wk = Linear(dim, dim)
        self.wv = Linear(dim, dim)
        self.wo = Linear(dim, dim)

    def forward(self, x):
        b, d = x.shape
        self.q = self.wq.forward(x).reshape(b, self.heads, self.hd)
        self.k = self.wk.forward(x).reshape(b, self.heads, self.hd)
        self.v = self.wv.forward(x).reshape(b, self.heads, self.hd)
        self.dots = np.einsum('bhd,bmd->bhm', self.q, self.k) * self.scale
        self.att = fast_softmax(self.dots)
        out = np.einsum('bhm,bmd->bhd', self.att, self.v)
        return self.wo.forward(out.reshape(b, d))

    def backward(self, dout):
        b, d = dout.shape
        d_out = self.wo.backward(dout).reshape(b, self.heads, self.hd)
        d_att = np.einsum('bhd,bmd->bhm', d_out, self.v)
        d_v = np.einsum('bhm,bhd->bmd', self.att, d_out)
        d_dots = self.att * (d_att - np.sum(self.att * d_att, axis=-1, keepdims=True)) * self.scale
        d_q = np.einsum('bhm,bmd->bhd', d_dots, self.k).reshape(b, d)
        d_k = np.einsum('bhm,bhd->bmd', d_dots, self.q).reshape(b, d)
        return self.wq.backward(d_q) + self.wk.backward(d_k) + self.wv.backward(d_v.reshape(b, d))

class SovereignBlock:
    def __init__(self, dim):
        self.ln1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim)
        self.ln2 = RMSNorm(dim)
        self.moe = RedundantMoE(dim)

    def forward(self, x):
        self.x = x
        self.n1 = self.ln1.forward(x)
        self.a = self.attn.forward(self.n1)
        self.x2 = x + self.a
        self.n2 = self.ln2.forward(self.x2)
        return self.x2 + self.moe.forward(self.n2)

    def backward(self, dout):
        dm = self.moe.backward(dout)
        dn2 = self.ln2.backward(dm)
        dx2 = dout + dn2
        da = self.attn.backward(dx2)
        dn1 = self.ln1.backward(da)
        return dx2 + dn1

class SovereignArchitect:
    def __init__(self, in_d, h_d, out_d, depth=2):
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
        def _collect(obj):
            p = []
            if isinstance(obj, (Linear, RMSNorm)): p.append(obj)
            elif hasattr(obj, '__dict__'):
                for v in obj.__dict__.values():
                    if isinstance(v, list): [p.extend(_collect(i)) for i in v]
                    else: p.extend(_collect(v))
            return p
        return _collect(self)

class Lion:
    def __init__(self, params, lr=1e-4, b1=0.9, b2=0.99, wd=0.01):
        self.params, self.lr, self.b1, self.b2, self.wd = params, lr, b1, b2, wd
        self.m = [np.zeros_like(p.W if hasattr(p, 'W') else p.g) for p in params]

    def step(self, scale=1.0):
        lr = self.lr * scale
        for i, p in enumerate(self.params):
            is_w = hasattr(p, 'W')
            w, g = (p.W, p.dW) if is_w else (p.g, p.dg)
            if self.wd > 0: w -= lr * self.wd * w
            u = np.sign(self.b1 * self.m[i] + (1.0 - self.b1) * g)
            w -= lr * u
            self.m[i] = self.b2 * self.m[i] + (1.0 - self.b2) * g
            if is_w: p.W = w
            else: p.g = w

def get_data(n, d, k):
    X = np.random.randn(n, d).astype(np.float32)
    y = np.random.randint(0, k, n)
    centers = np.random.randn(k, d).astype(np.float32) * 4.0
    X += centers[y]
    return (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-6), y

def train():
    N, D, K = 10000, 784, 10
    X, y = get_data(N, D, K)
    model = SovereignArchitect(D, 128, K, depth=2)
    p_list = model.params()
    opt = Lion(p_list, lr=1e-4, wd=0.01)
    bs, epochs = 64, 20
    
    print(f"OMEGA-ASI | SOVEREIGN-V4 | PARAMS: {sum(p.W.size if hasattr(p, 'W') else p.g.size for p in p_list)}")
    
    for ep in range(epochs):
        idx = np.random.permutation(N)
        l_acc, a_acc, t0 = 0, 0, time.time()
        sched = 0.5 * (1 + np.cos(np.pi * ep / epochs))
        
        for i in range(0, N, bs):
            bi = idx[i : i + bs]
            xb, yb = X[bi], y[bi]
            m = xb.shape[0]
            
            logits = model.forward(xb)
            probs = fast_softmax(logits)
            
            loss = -np.mean(np.log(probs[range(m), yb] + 1e-10))
            l_acc += loss * (m / N)
            a_acc += np.mean(np.argmax(probs, axis=1) == yb) * (m / N)
            
            dout = probs.copy()
            dout[range(m), yb] -= 1
            model.backward(dout / m)
            
            gn = np.sqrt(sum(np.sum((p.dW if hasattr(p, 'W') else p.dg)**2) for p in p_list))
            if gn > 1.0:
                for p in p_list:
                    if hasattr(p, 'W'): p.dW /= gn
                    else: p.dg /= gn
            opt.step(scale=sched)
            
        print(f"EP:{ep:02d} | LOSS:{l_acc:.4f} | ACC:{a_acc:.4f} | SPEED:{N/(time.time()-t0):.0f} samples/s")

if __name__ == "__main__":
    train()
