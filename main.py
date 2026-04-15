
import numpy as np

class FastOps:
    @staticmethod
    def swiglu(x):
        h = x.shape[-1] // 2
        a, b = x[..., :h], x[..., h:]
        return (a * (1.0 / (1.0 + np.exp(-np.clip(a, -10, 10))))) * b

    @staticmethod
    def d_swiglu(x, d):
        h = x.shape[-1] // 2
        a, b = x[..., :h], x[..., h:]
        s = 1.0 / (1.0 + np.exp(-np.clip(a, -10, 10)))
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
        dg = 0.5 * (1 + t) + 0.5 * a * (1 - t**2) * (0.79788 * (1 + 0.1341 * a**2))
        da = d * b * dg
        db = d * (0.5 * a * (1 + t))
        return np.concatenate([da, db], axis=-1)

    @staticmethod
    def softmax(x):
        m = np.max(x, axis=-1, keepdims=True)
        e = np.exp(x - m)
        return e / (np.sum(e, axis=-1, keepdims=True) + 1e-12)


class Linear:
    def __init__(self, i, o):
        limit = np.sqrt(6.0 / (i + o))
        self.W = np.random.uniform(-limit, limit, (i, o)).astype("f")
        self.b = np.zeros(o, "f")
        self.x, self.dW, self.db = None, None, None

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
        self.e, self.x, self.v = e, None, None

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
    def __init__(self, d, m=2048):
        f = 1.0 / (10000 ** (np.arange(0, d, 2) / d))
        t = np.arange(m)
        fr = np.outer(t, f)
        self.cos, self.sin = np.cos(fr), np.sin(fr)

    def apply(self, x, rev=False):
        b, s, h, d = x.shape
        d2 = d // 2
        c, sn = self.cos[:s][None, :, None, :], self.sin[:s][None, :, None, :]
        x1, x2 = x[..., :d2], x[..., d2:]
        if rev: return np.concatenate([x1 * c + x2 * sn, x2 * c - x1 * sn], axis=-1)
        return np.concatenate([x1 * c - x2 * sn, x2 * c + x1 * sn], axis=-1)


class SovereignAttention:
    def __init__(self, d, h=8, k=2):
        self.d, self.h, self.k, self.hd = d, h, k, d // h
        self.g = h // k
        self.wq = Linear(d, d)
        self.wk = Linear(d, k * self.hd)
        self.wv = Linear(d, k * self.hd)
        self.wo = Linear(d, d)
        self.rope, self.sc = RoPE(self.hd), self.hd**-0.5
        self.q, self.kc, self.vc, self.p = None, None, None, None

    def forward(self, x):
        b, s, _ = x.shape
        q = self.wq.forward(x).reshape(b, s, self.h, self.hd)
        k = self.wk.forward(x).reshape(b, s, self.k, self.hd)
        v = self.wv.forward(x).reshape(b, s, self.k, self.hd)
        self.q, self.kc, self.vc = self.rope.apply(q), self.rope.apply(k), v
        kr = np.repeat(self.kc, self.g, axis=2)
        vr = np.repeat(self.vc, self.g, axis=2)
        at = np.einsum("bshd,bthd->bsht", self.q, kr) * self.sc
        self.p = FastOps.softmax(at)
        out = np.einsum("bsht,bthd->bshd", self.p, vr).reshape(b, s, self.d)
        return self.wo.forward(out)

    def backward(self, d):
        b, s, _ = d.shape
        do = self.wo.backward(d).reshape(b, s, self.h, self.hd)
        kr = np.repeat(self.kc, self.g, axis=2)
        vr = np.repeat(self.vc, self.g, axis=2)
        dvr = np.einsum("bsht,bshd->bthd", self.p, do)
        dp = np.einsum("bshd,bthd->bsht", do, vr)
        da = self.p * (dp - np.sum(self.p * dp, axis=-1, keepdims=True)) * self.sc
        dq = self.rope.apply(np.einsum("bsht,bthd->bshd", da, kr), rev=True)
        dkr = np.einsum("bsht,bshd->bthd", da, self.q)
        dkc = self.rope.apply(dkr.reshape(b, s, self.k, self.g, self.hd).sum(axis=3), rev=True)
        dvc = dvr.reshape(b, s, self.k, self.g, self.hd).sum(axis=3)
        return self.wq.backward(dq.reshape(b, s, -1)) + self.wk.backward(dkc.reshape(b, s, -1)) + self.wv.backward(dvc.reshape(b, s, -1))


class SovereignMLP:
    def __init__(self, d, exp=4):
        self.up = Linear(d, d * exp * 2)
        self.dn = Linear(d * exp, d)
        self.gate = Linear(d, d * exp)
        self.x_up, self.x_gate, self.act = None, None, None

    def forward(self, x):
        self.x_up = self.up.forward(x)
        self.x_gate = np.tanh(self.gate.forward(x))
        self.act = FastOps.swiglu(self.x_up) * self.x_gate
        return self.dn.forward(self.act)

    def backward(self, d):
        d_dn = self.dn.backward(d)
        d_gate = d_dn * self.act * (1 - self.x_gate**2)
        d_act = d_dn * self.x_gate
        d_up = FastOps.d_swiglu(self.x_up, d_act)
        return self.up.backward(d_up) + self.gate.backward(d_gate)


class SovereignBlock:
    def __init__(self, d):
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn, self.mlp = SovereignAttention(d), SovereignMLP(d)

    def forward(self, x):
        x = x + self.attn.forward(self.n1.forward(x))
        return x + self.mlp.forward(self.n2.forward(x))

    def backward(self, d):
        dm = self.mlp.backward(d)
        d_res = d + self.n2.backward(dm)
        da = self.attn.backward(d_res)
        return d_res + self.n1.backward(da)


class OMEGA_ASI:
    def __init__(self, i=784, h=256, o=10, depth=4):
        self.stem = Linear(i, h)
        self.blocks = [SovereignBlock(h) for _ in range(depth)]
        self.norm = RMSNorm(h)
        self.head = Linear(h, o)

    def forward(self, x):
        if x.ndim == 2: x = x[:, None, :]
        x = self.stem.forward(x)
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
            elif hasattr(obj, "__dict__"): [find(v) for v in obj.__dict__.values() if v is not obj]
        find(self)
        return list(set(p))


class Lion:
    def __init__(self, params, lr=1e-4, b1=0.9, b2=0.99, wd=0.01):
        self.params, self.lr, self.b1, self.b2, self.wd = params, lr, b1, b2, wd
        self.m = {id(p): np.zeros_like(p.W) if hasattr(p, "W") else np.zeros_like(p.g) for p in params}
        self.mb = {id(p): np.zeros_like(p.b) if hasattr(p, "b") else None for p in params}

    def step(self, lr_scale=1.0):
        lr = self.lr * lr_scale
        for p in self.params:
            pid = id(p)
            if hasattr(p, "W"):
                for attr, mom in [( "W", self.m), ( "b", self.mb)]:
                    if mom[pid] is None: continue
                    g, w = getattr(p, "d" + attr), getattr(p, attr)
                    u = np.sign(self.b1 * mom[pid] + (1.0 - self.b1) * g)
                    w -= lr * (u + self.wd * w if attr == "W" else u)
                    mom[pid] = self.b2 * mom[pid] + (1.0 - self.b2) * g
                    setattr(p, attr, w)
            else:
                u = np.sign(self.b1 * self.m[pid] + (1.0 - self.b1) * p.dg)
                p.g -= lr * (u + self.wd * p.g)
                self.m[pid] = self.b2 * self.m[pid] + (1.0 - self.b2) * p.dg


def main():
    N, D, C = 1024, 784, 10
    X = np.random.randn(N, D).astype("f")
    Y = np.random.randint(0, C, N)
    model = OMEGA_ASI(i=D, h=128, o=C, depth=2)
    params = model.get_params()
    opt = Lion(params, lr=5e-4, wd=0.01)
    bs, epochs = 32, 30
    for epoch in range(epochs):
        idx = np.random.permutation(N)
        l_sum, a_sum = 0, 0
        lr_s = 0.5 * (1 + np.cos(np.pi * epoch / epochs))
        for i in range(0, N, bs):
            xb, yb = X[idx[i:i+bs]], Y[idx[i:i+bs]]
            probs = FastOps.softmax(model.forward(xb))
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
            opt.step(lr_s)
        print(f"EP {epoch+1:02d} | LOSS: {l_sum/N:.4f} | ACC: {a_sum/N:.4f} | LR: {opt.lr*lr_s:.6f}")


if __name__ == "__main__":
    main()
