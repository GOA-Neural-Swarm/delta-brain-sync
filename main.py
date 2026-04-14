import numpy as np


def swiglu(x):
    h = x.shape[-1] // 2
    a, b = x[..., :h], x[..., h:]
    s = 1 / (1 + np.exp(-np.clip(a, -12, 12)))
    return (a * s) * b


def d_swiglu(x, d):
    h = x.shape[-1] // 2
    a, b = x[..., :h], x[..., h:]
    s = 1 / (1 + np.exp(-np.clip(a, -12, 12)))
    sw = a * s
    da = d * b * (s + sw * (1 - s))
    db = d * sw
    return np.concatenate([da, db], axis=-1)


class Linear:
    def __init__(self, i, o, s=1.0):
        self.W = np.random.randn(i, o).astype("f") * (np.sqrt(2 / i) * s)
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
        self.g, self.e = np.ones(d, "f"), e
        self.x, self.v = None, None

    def forward(self, x):
        self.x = x
        self.v = 1 / np.sqrt((x**2).mean(-1, keepdims=True) + self.e)
        return self.g * (x * self.v)

    def backward(self, d):
        nx = self.x * self.v
        self.dg = (d * nx).sum(axis=tuple(range(d.ndim - 1)))
        dn = d * self.g
        return self.v * (dn - nx * (dn * nx).mean(-1, keepdims=True))


class RoPE:
    def __init__(self, d, m=4096):
        f = 1 / (10000 ** (np.arange(0, d, 2) / d))
        t = np.arange(m)
        fr = np.outer(t, f)
        self.c, self.s = np.cos(fr), np.sin(fr)

    def apply(self, x, rev=False):
        b, s, h, d = x.shape
        d2 = d // 2
        x1, x2 = x[..., :d2], x[..., d2:]
        c, sn = self.c[:s][None, :, None, :], self.s[:s][None, :, None, :]
        if rev:
            return np.concatenate([x1 * c + x2 * sn, x2 * c - x1 * sn], -1)
        return np.concatenate([x1 * c - x2 * sn, x2 * c + x1 * sn], -1)


class GQA:
    def __init__(self, d, h=8, k=2):
        self.d, self.h, self.k, self.hd = d, h, k, d // h
        self.g = h // k
        self.wq, self.wk, self.wv, self.wo = (
            Linear(d, d),
            Linear(d, k * self.hd),
            Linear(d, k * self.hd),
            Linear(d, d),
        )
        self.rope, self.sc = RoPE(self.hd), (d // h) ** -0.5
        self.q, self.k_cache, self.v_cache, self.p = None, None, None, None

    def forward(self, x):
        b, s, _ = x.shape
        q = self.wq.forward(x).reshape(b, s, self.h, self.hd)
        k = self.wk.forward(x).reshape(b, s, self.k, self.hd)
        v = self.wv.forward(x).reshape(b, s, self.k, self.hd)
        self.q, self.k_cache = self.rope.apply(q), self.rope.apply(k)
        self.v_cache = v
        kr, vr = np.repeat(self.k_cache, self.g, 2), np.repeat(v, self.g, 2)
        at = np.einsum("bshd,bthd->bsht", self.q, kr) * self.sc
        ex = np.exp(at - np.max(at, axis=-1, keepdims=True))
        self.p = ex / (np.sum(ex, axis=-1, keepdims=True) + 1e-10)
        o = np.einsum("bsht,bthd->bshd", self.p, vr)
        return self.wo.forward(o.reshape(b, s, self.d))

    def backward(self, d):
        b, s, _ = d.shape
        do = self.wo.backward(d).reshape(b, s, self.h, self.hd)
        kr, vr = np.repeat(self.k_cache, self.g, 2), np.repeat(self.v_cache, self.g, 2)
        dvr = np.einsum("bsht,bshd->bthd", self.p, do)
        dp = np.einsum("bshd,bthd->bsht", do, vr)
        da = self.p * (dp - (self.p * dp).sum(-1, keepdims=True)) * self.sc
        dq = self.rope.apply(np.einsum("bsht,bthd->bshd", da, kr), True)
        dkr = np.einsum("bsht,bshd->bthd", da, self.q)
        dk = self.rope.apply(dkr.reshape(b, s, self.k, self.g, self.hd).sum(3), True)
        dv = dvr.reshape(b, s, self.k, self.g, self.hd).sum(3)
        return (
            self.wq.backward(dq.reshape(b, s, -1))
            + self.wk.backward(dk.reshape(b, s, -1))
            + self.wv.backward(dv.reshape(b, s, -1))
        )


class RedundantLogic:
    def __init__(self, d):
        self.gemini = [Linear(d, d * 2), Linear(d * 2, d)]
        self.groq = [Linear(d, d * 2), Linear(d * 2, d)]
        self.gate = Linear(d, 2)
        self.h_gem, self.h_gro, self.p = None, None, None

    def forward(self, x):
        self.h_gem = swiglu(self.gemini[0].forward(x))
        o_gem = self.gemini[1].forward(self.h_gem)
        self.h_gro = swiglu(self.groq[0].forward(x))
        o_gro = self.groq[1].forward(self.h_gro)
        g = self.gate.forward(x)
        ex = np.exp(g - g.max(-1, keepdims=True))
        self.p = ex / (ex.sum(-1, keepdims=True) + 1e-10)
        self.o_gem, self.o_gro = o_gem, o_gro
        return self.p[..., :1] * o_gem + self.p[..., 1:2] * o_gro

    def backward(self, d):
        dp = np.stack([(d * self.o_gem).sum(-1), (d * self.o_gro).sum(-1)], -1)
        dg = self.p * (dp - (self.p * dp).sum(-1, keepdims=True))
        dx = self.gate.backward(dg)
        d_gem = self.gemini[1].backward(d * self.p[..., :1])
        dx += self.gemini[0].backward(d_swiglu(self.gemini[0].x, d_gem))
        d_gro = self.groq[1].backward(d * self.p[..., 1:2])
        dx += self.groq[0].backward(d_swiglu(self.groq[0].x, d_gro))
        return dx


class MoE:
    def __init__(self, d, n=4):
        self.d, self.n = d, n
        self.gate = Linear(d, n)
        self.experts = [[Linear(d, d * 2), Linear(d * 2, d)] for _ in range(n)]
        self.p, self.sel, self.cache = None, None, []

    def forward(self, x):
        s = x.shape
        x_flat = x.reshape(-1, self.d)
        g = self.gate.forward(x_flat)
        ex = np.exp(g - g.max(-1, keepdims=True))
        self.p = ex / (ex.sum(-1, keepdims=True) + 1e-10)
        self.sel = np.argmax(self.p, -1)
        out = np.zeros_like(x_flat)
        self.cache = []
        for i in range(self.n):
            m = self.sel == i
            if not np.any(m):
                self.cache.append(None)
                continue
            h = swiglu(self.experts[i][0].forward(x_flat[m]))
            y = self.experts[i][1].forward(h)
            out[m] = y * self.p[m, i : i + 1]
            self.cache.append((m, h, y))
        return out.reshape(s)

    def backward(self, d):
        s = d.shape
        d_flat = d.reshape(-1, self.d)
        dx, dp = np.zeros_like(d_flat), np.zeros_like(self.p)
        for i in range(self.n):
            if self.cache[i] is None:
                continue
            m, h, y = self.cache[i]
            dp[m, i] = (d_flat[m] * y).sum(-1)
            dy = d_flat[m] * self.p[m, i : i + 1]
            dh = self.experts[i][1].backward(dy)
            dx[m] += self.experts[i][0].backward(d_swiglu(self.experts[i][0].x, dh))
        dg = self.p * (dp - (self.p * dp).sum(-1, keepdims=True))
        return (dx + self.gate.backward(dg)).reshape(s)


class Block:
    def __init__(self, d):
        self.n1, self.attn = RMSNorm(d), GQA(d)
        self.n2, self.logic = RMSNorm(d), RedundantLogic(d)
        self.n3, self.moe = RMSNorm(d), MoE(d)

    def forward(self, x):
        x = x + self.attn.forward(self.n1.forward(x))
        x = x + self.logic.forward(self.n2.forward(x))
        x = x + self.moe.forward(self.n3.forward(x))
        return x

    def backward(self, d):
        d = d + self.n3.backward(self.moe.backward(d))
        d = d + self.n2.backward(self.logic.backward(d))
        d = d + self.n1.backward(self.attn.backward(d))
        return d


class OMEGA_ASI:
    def __init__(self, i=784, h=128, o=10, depth=2):
        self.stem = Linear(i, h)
        self.blocks = [Block(h) for _ in range(depth)]
        self.norm = RMSNorm(h)
        self.head = Linear(h, o)

    def forward(self, x):
        x = self.stem.forward(x)[:, None, :]
        for b in self.blocks:
            x = b.forward(x)
        self.feat = self.norm.forward(x[:, 0, :])
        return self.head.forward(self.feat)

    def backward(self, d):
        d = self.norm.backward(self.head.backward(d))[:, None, :]
        for b in reversed(self.blocks):
            d = b.backward(d)
        self.stem.backward(d[:, 0, :])

    def get_params(self):
        p = []

        def find(obj):
            if isinstance(obj, (Linear, RMSNorm)):
                p.append(obj)
            elif isinstance(obj, list):
                [find(i) for i in obj]
            elif hasattr(obj, "__dict__"):
                [find(v) for v in obj.__dict__.values()]

        find(self)
        return p


class Lion:
    def __init__(self, params, lr=1e-4, b1=0.9, b2=0.99, wd=0.01):
        self.params, self.lr, self.b1, self.b2, self.wd = params, lr, b1, b2, wd
        self.m = [np.zeros_like(getattr(p, "W", getattr(p, "g", 0))) for p in params]
        self.mb = [np.zeros_like(p.b) if hasattr(p, "b") else None for p in params]

    def step(self):
        for i, p in enumerate(self.params):
            if hasattr(p, "W"):
                for attr, mom in [("W", self.m), ("b", self.mb)]:
                    if mom[i] is None:
                        continue
                    g, w = getattr(p, "d" + attr), getattr(p, attr)
                    u = np.sign(self.b1 * mom[i] + (1 - self.b1) * g)
                    w -= self.lr * (u + self.wd * w if attr == "W" else u)
                    mom[i] = self.b2 * mom[i] + (1 - self.b2) * g
                    setattr(p, attr, w)
            else:
                u = np.sign(self.b1 * self.m[i] + (1 - self.b1) * p.dg)
                p.g -= self.lr * (u + self.wd * p.g)
                self.m[i] = self.b2 * self.m[i] + (1 - self.b2) * p.dg


def train():
    N, D, C = 1024, 784, 10
    X = np.random.randn(N, D).astype("f")
    Y = np.random.randint(0, C, N)
    model = OMEGA_ASI(D, 64, C, 1)
    optimizer = Lion(model.get_params(), lr=3e-4)

    for epoch in range(20):
        idx = np.random.permutation(N)
        t_loss, t_acc = 0, 0
        for i in range(0, N, 32):
            xb, yb = X[idx[i : i + 32]], Y[idx[i : i + 32]]
            logits = model.forward(xb)
            probs = np.exp(logits - logits.max(1, keepdims=True))
            probs /= probs.sum(1, keepdims=True) + 1e-10

            t_loss += -np.log(probs[range(len(yb)), yb] + 1e-10).mean() * len(yb)
            t_acc += (probs.argmax(1) == yb).sum()

            dout = probs.copy()
            dout[range(len(yb)), yb] -= 1
            model.backward(dout / len(yb))

            gn = np.sqrt(
                sum(
                    (getattr(p, "dW", 0) ** 2).sum()
                    + (getattr(p, "db", 0) ** 2).sum()
                    + (getattr(p, "dg", 0) ** 2).sum()
                    for p in model.get_params()
                )
            )
            if gn > 1.0:
                for p in model.get_params():
                    if hasattr(p, "dW"):
                        p.dW *= 1.0 / gn
                        p.db *= 1.0 / gn
                    if hasattr(p, "dg"):
                        p.dg *= 1.0 / gn
            optimizer.step()

        print(f"Epoch {epoch+1} | Loss: {t_loss/N:.4f} | Acc: {t_acc/N:.4f}")


if __name__ == "__main__":
    train()
