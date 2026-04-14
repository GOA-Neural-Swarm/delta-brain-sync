import numpy as np

class Ops:
    @staticmethod
    def swiglu(x):
        h = x.shape[-1] // 2
        a, b = x[..., :h], x[..., h:]
        s = 1.0 / (1.0 + np.exp(-np.clip(a, -12, 12)))
        return (a * s) * b

    @staticmethod
    def d_swiglu(x, d):
        h = x.shape[-1] // 2
        a, b = x[..., :h], x[..., h:]
        s = 1.0 / (1.0 + np.exp(-np.clip(a, -12, 12)))
        sw = a * s
        da = d * b * (s + sw * (1.0 - s))
        db = d * sw
        return np.concatenate([da, db], axis=-1)

    @staticmethod
    def geglu(x):
        h = x.shape[-1] // 2
        a, b = x[..., :h], x[..., h:]
        g = 0.5 * a * (1 + np.tanh(0.79788 * (a + 0.0447 * a**3)))
        return g * b

    @staticmethod
    def d_geglu(x, d):
        h = x.shape[-1] // 2
        a, b = x[..., :h], x[..., h:]
        t = np.tanh(0.79788 * (a + 0.0447 * a**3))
        dg = 0.5 * (1 + t) + (0.39894 * a * np.exp(-0.5 * a**2))
        da = d * b * dg
        db = d * (0.5 * a * (1 + t))
        return np.concatenate([da, db], axis=-1)

    @staticmethod
    def softmax(x, axis=-1):
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e / (np.sum(e, axis=axis, keepdims=True) + 1e-10)

class Linear:
    def __init__(self, i, o, std=None):
        self.W = np.random.randn(i, o).astype("f") * (std or (2/i)**0.5)
        self.b = np.zeros(o, "f")
        self.dW, self.db = np.zeros_like(self.W), np.zeros_like(self.b)
        self.x = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, d):
        self.dW = self.x.reshape(-1, self.x.shape[-1]).T @ d.reshape(-1, d.shape[-1])
        self.db = d.sum(axis=tuple(range(d.ndim - 1)))
        return d @ self.W.T

class RMSNorm:
    def __init__(self, d, e=1e-6):
        self.g = np.ones(d, "f")
        self.e = e
        self.x, self.v = None, None

    def forward(self, x):
        self.x = x
        self.v = 1.0 / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.e)
        return self.g * (x * self.v)

    def backward(self, d):
        nx = self.x * self.v
        self.dg = (d * nx).sum(axis=tuple(range(d.ndim - 1)))
        dn = d * self.g
        return self.v * (dn - nx * np.mean(dn * nx, axis=-1, keepdims=True))

class RoPE:
    def __init__(self, d, m=4096):
        f = 1.0 / (10000 ** (np.arange(0, d, 2) / d))
        fr = np.outer(np.arange(m), f)
        self.cos, self.sin = np.cos(fr), np.sin(fr)

    def apply(self, x, rev=False):
        b, s, h, d = x.shape
        d2 = d // 2
        c, sn = self.cos[:s][None, :, None, :], self.sin[:s][None, :, None, :]
        x1, x2 = x[..., :d2], x[..., d2:]
        if rev:
            return np.concatenate([x1 * c + x2 * sn, x2 * c - x1 * sn], axis=-1)
        return np.concatenate([x1 * c - x2 * sn, x2 * c + x1 * sn], axis=-1)

class GQA:
    def __init__(self, d, h=8, k=2):
        self.d, self.h, self.k, self.hd = d, h, k, d // h
        self.g = h // k
        self.wq, self.wk, self.wv, self.wo = Linear(d, d), Linear(d, k*self.hd), Linear(d, k*self.hd), Linear(d, d)
        self.rope, self.sc = RoPE(self.hd), self.hd**-0.5
        self.q, self.kc, self.vc, self.p = None, None, None, None

    def forward(self, x):
        b, s, _ = x.shape
        q = self.wq.forward(x).reshape(b, s, self.h, self.hd)
        k = self.wk.forward(x).reshape(b, s, self.k, self.hd)
        v = self.wv.forward(x).reshape(b, s, self.k, self.hd)
        self.q, self.kc, self.vc = self.rope.apply(q), self.rope.apply(k), v
        kr, vr = np.repeat(self.kc, self.g, axis=2), np.repeat(self.vc, self.g, axis=2)
        at = np.einsum("bshd,bthd->bsht", self.q, kr) * self.sc
        self.p = Ops.softmax(at)
        out = np.einsum("bsht,bthd->bshd", self.p, vr).reshape(b, s, self.d)
        return self.wo.forward(out)

    def backward(self, d):
        b, s, _ = d.shape
        do = self.wo.backward(d).reshape(b, s, self.h, self.hd)
        kr, vr = np.repeat(self.kc, self.g, axis=2), np.repeat(self.vc, self.g, axis=2)
        dvr = np.einsum("bsht,bshd->bthd", self.p, do)
        dp = np.einsum("bshd,bthd->bsht", do, vr)
        da = self.p * (dp - (self.p * dp).sum(axis=-1, keepdims=True)) * self.sc
        dq = self.rope.apply(np.einsum("bsht,bthd->bshd", da, kr), rev=True)
        dkr = np.einsum("bsht,bshd->bthd", da, self.q)
        dkc = self.rope.apply(dkr.reshape(b, s, self.k, self.g, self.hd).sum(axis=3), rev=True)
        dvc = dvr.reshape(b, s, self.k, self.g, self.hd).sum(axis=3)
        return self.wq.backward(dq.reshape(b, s, -1)) + self.wk.backward(dkc.reshape(b, s, -1)) + self.wv.backward(dvc.reshape(b, s, -1))

class RedundantLogic:
    def __init__(self, d):
        self.gate = Linear(d, 2)
        self.gemini = [Linear(d, d*2), Linear(d, d)]
        self.groq = [Linear(d, d*2), Linear(d, d)]
        self.p, self.o_gem, self.o_gro = None, None, None

    def forward(self, x):
        self.h_gem = Ops.swiglu(self.gemini[0].forward(x))
        self.o_gem = self.gemini[1].forward(self.h_gem)
        self.h_gro = Ops.geglu(self.groq[0].forward(x))
        self.o_gro = self.groq[1].forward(self.h_gro)
        self.p = Ops.softmax(self.gate.forward(x))
        return self.p[..., 0:1] * self.o_gem + self.p[..., 1:2] * self.o_gro

    def backward(self, d):
        dp = np.stack([(d * self.o_gem).sum(axis=-1), (d * self.o_gro).sum(axis=-1)], axis=-1)
        dg = self.p * (dp - (self.p * dp).sum(axis=-1, keepdims=True))
        dx = self.gate.backward(dg)
        d_gem = self.gemini[1].backward(d * self.p[..., 0:1])
        dx += self.gemini[0].backward(Ops.d_swiglu(self.gemini[0].x, d_gem))
        d_gro = self.groq[1].backward(d * self.p[..., 1:2])
        dx += self.groq[0].backward(Ops.d_geglu(self.groq[0].x, d_gro))
        return dx

class Block:
    def __init__(self, d):
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn = GQA(d)
        self.logic = RedundantLogic(d)

    def forward(self, x):
        x = x + self.attn.forward(self.n1.forward(x))
        x = x + self.logic.forward(self.n2.forward(x))
        return x

    def backward(self, d):
        d_logic = self.logic.backward(d)
        d = d + self.n2.backward(d_logic)
        d_attn = self.attn.backward(d)
        d = d + self.n1.backward(d_attn)
        return d

class OMEGA_ASI:
    def __init__(self, i=784, h=128, o=10, depth=2):
        self.stem = Linear(i, h)
        self.blocks = [Block(h) for _ in range(depth)]
        self.norm = RMSNorm(h)
        self.head = Linear(h, o)

    def forward(self, x):
        x = self.stem.forward(x)[:, None, :]
        for b in self.blocks: x = b.forward(x)
        self.feat = self.norm.forward(x[:, 0, :])
        return self.head.forward(self.feat)

    def backward(self, d):
        d = self.norm.backward(self.head.backward(d))[:, None, :]
        for b in reversed(self.blocks): d = b.backward(d)
        self.stem.backward(d[:, 0, :])

    def get_params(self):
        p = []
        def find(obj):
            if isinstance(obj, (Linear, RMSNorm)): p.append(obj)
            elif isinstance(obj, list): [find(i) for i in obj]
            elif hasattr(obj, "__dict__"): [find(v) for v in obj.__dict__.values()]
        find(self)
        return p

class Lion:
    def __init__(self, params, lr=1e-4, b1=0.9, b2=0.99, wd=0.01):
        self.params, self.lr, self.b1, self.b2, self.wd = params, lr, b1, b2, wd
        self.m = [np.zeros_like(p.W if hasattr(p, "W") else p.g) for p in params]
        self.mb = [np.zeros_like(p.b) if hasattr(p, "b") else None for p in params]

    def step(self):
        for i, p in enumerate(self.params):
            if hasattr(p, "W"):
                for attr, mom in [("W", self.m), ("b", self.mb)]:
                    if mom[i] is None: continue
                    g, w = getattr(p, "d" + attr), getattr(p, attr)
                    u = np.sign(self.b1 * mom[i] + (1.0 - self.b1) * g)
                    w -= self.lr * (u + self.wd * w if attr == "W" else u)
                    mom[i] = self.b2 * mom[i] + (1.0 - self.b2) * g
                    setattr(p, attr, w)
            else:
                u = np.sign(self.b1 * self.m[i] + (1.0 - self.b1) * p.dg)
                p.g -= self.lr * (u + self.wd * p.g)
                self.m[i] = self.b2 * self.m[i] + (1.0 - self.b2) * p.dg

def train():
    N, D, C = 1024, 784, 10
    X = np.random.randn(N, D).astype("f")
    Y = np.random.randint(0, C, N)
    model = OMEGA_ASI(D, 64, C, 1)
    params = model.get_params()
    opt = Lion(params, lr=1e-4)
    bs = 32
    for epoch in range(100):
        idx = np.random.permutation(N)
        l_sum, a_sum = 0, 0
        for i in range(0, N, bs):
            xb, yb = X[idx[i:i+bs]], Y[idx[i:i+bs]]
            logits = model.forward(xb)
            probs = Ops.softmax(logits)
            l_sum += -np.log(probs[range(len(yb)), yb] + 1e-10).sum()
            a_sum += (probs.argmax(1) == yb).sum()
            dout = probs.copy()
            dout[range(len(yb)), yb] -= 1
            model.backward(dout / len(yb))
            gn = np.sqrt(sum((getattr(p, "dW", 0)**2).sum() + (getattr(p, "db", 0)**2).sum() + (getattr(p, "dg", 0)**2).sum() for p in params))
            if gn > 1.0:
                for p in params:
                    if hasattr(p, "dW"): p.dW /= gn; p.db /= gn
                    if hasattr(p, "dg"): p.dg /= gn
            opt.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1} | Loss: {l_sum/N:.4f} | Acc: {a_sum/N:.4f}")

if __name__ == "__main__":
    train()
