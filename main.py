
import numpy as np
import time

def fast_softmax(x, axis=-1):
    c = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - c)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)

class Linear:
    def __init__(self, in_f, out_f, scale=1.0):
        limit = np.sqrt(6.0 / (in_f + out_f)) * scale
        self.W = np.random.uniform(-limit, limit, (in_f, out_f)).astype(np.float32)
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
        self.x, self.rstd = None, None

    def forward(self, x):
        self.x = x
        self.rstd = 1.0 / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return self.g * (x * self.rstd)

    def backward(self, dout):
        nx = self.x * self.rstd
        self.dg = np.sum(dout * nx, axis=0)
        v = (dout * self.g)
        return self.rstd * (v - nx * np.mean(v * nx, axis=-1, keepdims=True))

class RedundantMoE:
    def __init__(self, dim):
        self.w1 = Linear(dim, dim * 2)
        self.w2 = Linear(dim, dim * 2)
        self.w3 = Linear(dim * 2, dim)
        self.gate = Linear(dim, 2)

    def forward(self, x):
        self.probs = fast_softmax(self.gate.forward(x))
        self.z1 = self.w1.forward(x)
        self.z2 = self.w2.forward(x)
        self.sig = 1.0 / (1.0 + np.exp(-np.clip(self.z1, -10, 10)))
        self.swish = self.z1 * self.sig
        self.o1 = self.w3.forward(self.swish * self.z2)
        self.o2 = self.w3.forward(self.z1 * self.z2)
        return self.probs[:, 0:1] * self.o1 + self.probs[:, 1:2] * self.o2

    def backward(self, dout):
        p1, p2 = self.probs[:, 0:1], self.probs[:, 1:2]
        dx = self.w1.backward(dout * p1) + self.w2.backward(dout * p2)
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
        b, s, d = x.shape
        self.q = self.wq.forward(x.reshape(-1, d)).reshape(b, s, self.heads, self.hd).transpose(0, 2, 1, 3)
        self.k = self.wk.forward(x.reshape(-1, d)).reshape(b, s, self.heads, self.hd).transpose(0, 2, 1, 3)
        self.v = self.wv.forward(x.reshape(-1, d)).reshape(b, s, self.heads, self.hd).transpose(0, 2, 1, 3)
        self.dots = np.matmul(self.q, self.k.transpose(0, 1, 3, 2)) * self.scale
        self.att = fast_softmax(self.dots)
        out = np.matmul(self.att, self.v).transpose(0, 2, 1, 3).reshape(b, s, d)
        return self.wo.forward(out.reshape(-1, d)).reshape(b, s, d)

    def backward(self, dout):
        b, s, d = dout.shape
        dout_reshaped = dout.reshape(-1, d)
        d_wo = self.wo.backward(dout_reshaped).reshape(b, s, self.heads, self.hd).transpose(0, 2, 1, 3)
        d_att = np.matmul(d_wo, self.v.transpose(0, 1, 3, 2))
        d_v = np.matmul(self.att.transpose(0, 1, 3, 2), d_wo)
        d_dots = self.att * (d_att - np.sum(self.att * d_att, axis=-1, keepdims=True)) * self.scale
        d_q = np.matmul(d_dots, self.k)
        d_k = np.matmul(d_dots.transpose(0, 1, 3, 2), self.q)
        dq = self.wq.backward(d_q.transpose(0, 2, 1, 3).reshape(-1, d))
        dk = self.wk.backward(d_k.transpose(0, 2, 1, 3).reshape(-1, d))
        dv = self.wv.backward(d_v.transpose(0, 2, 1, 3).reshape(-1, d))
        return (dq + dk + dv).reshape(b, s, d)

class SovereignBlock:
    def __init__(self, dim):
        self.ln1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim)
        self.ln2 = RMSNorm(dim)
        self.moe = RedundantMoE(dim)
        self.ls1 = np.ones(dim, dtype=np.float32) * 1e-2
        self.ls2 = np.ones(dim, dtype=np.float32) * 1e-2

    def forward(self, x):
        self.x = x
        n1 = self.ln1.forward(x)
        self.a = self.attn.forward(n1)
        self.x2 = x + self.ls1 * self.a
        n2 = self.ln2.forward(self.x2)
        self.m = self.moe.forward(n2.reshape(-1, n2.shape[-1])).reshape(n2.shape)
        return self.x2 + self.ls2 * self.m

    def backward(self, dout):
        self.dls2 = np.sum(dout * self.m, axis=(0, 1))
        dm = self.moe.backward((dout * self.ls2).reshape(-1, dout.shape[-1])).reshape(dout.shape)
        dn2 = self.ln2.backward(dm)
        dx2 = dout + dn2
        self.dls1 = np.sum(dx2 * self.a, axis=(0, 1))
        da = self.attn.backward(dx2 * self.ls1)
        dn1 = self.ln1.backward(da)
        return dx2 + dn1

class SovereignArchitect:
    def __init__(self, in_d, h_d, out_d, depth=2):
        self.patch_size = 4
        self.num_patches = (28 // self.patch_size) ** 2
        self.patch_dim = self.patch_size ** 2
        self.stem = Linear(self.patch_dim, h_d)
        self.blocks = [SovereignBlock(h_d) for _ in range(depth)]
        self.norm = RMSNorm(h_d)
        self.head = Linear(h_d, out_d, scale=0.1)

    def forward(self, x):
        b = x.shape[0]
        x = x.reshape(b, self.num_patches, self.patch_dim)
        x = self.stem.forward(x.reshape(-1, self.patch_dim)).reshape(b, self.num_patches, -1)
        for b_block in self.blocks: x = b_block.forward(x)
        x_pool = np.mean(x, axis=1)
        return self.head.forward(self.norm.forward(x_pool))

    def backward(self, dout):
        dout = self.norm.backward(self.head.backward(dout))
        b = dout.shape[0]
        dout = np.tile(dout[:, np.newaxis, :] / self.num_patches, (1, self.num_patches, 1))
        for b_block in reversed(self.blocks): dout = b_block.backward(dout)
        self.stem.backward(dout.reshape(-1, dout.shape[-1]))

    def params(self):
        def _collect(obj):
            p = []
            if isinstance(obj, (Linear, RMSNorm)): p.append(obj)
            elif hasattr(obj, 'ls1'): p.append(obj) 
            elif hasattr(obj, '__dict__'):
                for v in obj.__dict__.values():
                    if isinstance(v, list): [p.extend(_collect(i)) for i in v]
                    else: p.extend(_collect(v))
            return p
        return _collect(self)

class Lion:
    def __init__(self, params, lr=1e-4, b1=0.9, b2=0.99, wd=0.01):
        self.params, self.lr, self.b1, self.b2, self.wd = params, lr, b1, b2, wd
        self.m = []
        for p in params:
            if hasattr(p, 'W'): self.m.append(np.zeros_like(p.W))
            elif hasattr(p, 'g'): self.m.append(np.zeros_like(p.g))
            else: self.m.append(np.zeros_like(p.ls1))

    def step(self, scale=1.0):
        lr = self.lr * scale
        for i, p in enumerate(self.params):
            if hasattr(p, 'W'):
                w, g = p.W, p.dW
                if self.wd > 0: w -= lr * self.wd * w
                u = np.sign(self.b1 * self.m[i] + (1.0 - self.b1) * g)
                w -= lr * u
                self.m[i] = self.b2 * self.m[i] + (1.0 - self.b2) * g
                p.W = w
                p.b -= lr * np.sign(p.db) * 0.1
            elif hasattr(p, 'g'):
                w, g = p.g, p.dg
                u = np.sign(self.b1 * self.m[i] + (1.0 - self.b1) * g)
                p.g -= lr * u
                self.m[i] = self.b2 * self.m[i] + (1.0 - self.b2) * g
            elif hasattr(p, 'ls1'):
                for attr in ['ls1', 'ls2']:
                    ls = getattr(p, attr)
                    grad = getattr(p, 'd' + attr)
                    ls -= lr * np.sign(grad)
                    setattr(p, attr, ls)

def get_data(n, d, k):
    X = np.random.randn(n, d).astype(np.float32)
    y = np.random.randint(0, k, n)
    centers = np.random.randn(k, d).astype(np.float32) * 4.0
    X += centers[y]
    X = (X - np.mean(X)) / (np.std(X) + 1e-6)
    return X, y

def train():
    N, D, K = 5000, 784, 10
    X, y = get_data(N, D, K)
    model = SovereignArchitect(D, 64, K, depth=2)
    p_list = model.params()
    opt = Lion(p_list, lr=1e-4, wd=0.01)
    bs, epochs = 64, 40

    print(f"OMEGA-ASI | V6-EVOLVED | PARAMS: {sum(p.W.size if hasattr(p, 'W') else p.g.size if hasattr(p, 'g') else p.ls1.size for p in p_list)}")

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

            gn = np.sqrt(sum(np.sum((p.dW if hasattr(p, 'W') else p.dg if hasattr(p, 'g') else 0)**2) for p in p_list))
            if gn > 1.0:
                for p in p_list:
                    if hasattr(p, 'W'): p.dW /= gn
                    elif hasattr(p, 'g'): p.dg /= gn
            opt.step(scale=sched)

        print(f"EP:{ep:03d} | LOSS:{l_acc:.4f} | ACC:{a_acc:.4f} | SPEED:{N/(time.time()-t0):.0f} samples/s")

if __name__ == "__main__":
    train()
