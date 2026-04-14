import numpy as np


def swi(x):
    h = x.shape[-1] // 2
    a, b = x[..., :h], x[..., h:]
    return a * (1 / (1 + np.exp(-np.clip(a, -10, 10)))) * b


def dswi(x, d):
    h = x.shape[-1] // 2
    a, b = x[..., :h], x[..., h:]
    s = 1 / (1 + np.exp(-np.clip(a, -10, 10)))
    sw = a * s
    return np.concatenate([d * b * (s + sw * (1 - s)), d * sw], -1)


class Linear:
    def __init__(self, i, o, s=1.0):
        self.W = np.random.randn(i, o).astype("f") * (np.sqrt(2 / i) * s)
        self.b = np.zeros(o, "f")

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, d):
        self.dW = self.x.reshape(-1, self.x.shape[-1]).T @ d.reshape(-1, d.shape[-1])
        self.db = d.sum(axis=tuple(range(d.ndim - 1)))
        return d @ self.W.T


class RMS:
    def __init__(self, d, e=1e-6):
        self.g, self.e = np.ones(d, "f"), e

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
    def __init__(self, d, m=2048):
        f = 1 / (10000 ** (np.arange(0, d, 2) / d))
        t = np.arange(m)
        fr = np.outer(t, f)
        self.c, self.s = np.cos(fr)[None, :, None, :], np.sin(fr)[None, :, None, :]

    def apply(self, x, r=False):
        b, s, h, d = x.shape
        d2 = d // 2
        x1, x2 = x[..., :d2], x[..., d2:]
        c, sn = self.c[:, :s], self.s[:, :s]
        return (
            np.concatenate([x1 * c + x2 * sn, x2 * c - x1 * sn], -1)
            if r
            else np.concatenate([x1 * c - x2 * sn, x2 * c + x1 * sn], -1)
        )


class Attn:
    def __init__(self, d, h=8, k=4):
        self.d, self.h, self.k, self.hd = d, h, k, d // h
        self.g = h // k
        self.wq, self.wk, self.wv, self.wo = (
            Linear(d, d),
            Linear(d, k * self.hd),
            Linear(d, k * self.hd),
            Linear(d, d),
        )
        self.rope, self.sc = RoPE(self.hd), (d // h) ** -0.5

    def forward(self, x):
        b, s, _ = x.shape
        q = self.wq.forward(x).reshape(b, s, self.h, self.hd)
        k = self.wk.forward(x).reshape(b, s, self.k, self.hd)
        v = self.wv.forward(x).reshape(b, s, self.k, self.hd)
        self.q, self.k, self.v = self.rope.apply(q), self.rope.apply(k), v
        kr, vr = np.repeat(self.k, self.g, 2), np.repeat(v, self.g, 2)
        at = np.einsum("bshd,bthd->bsht", self.q, kr) * self.sc
        ex = np.exp(at - np.max(at, axis=-1, keepdims=True))
        self.p = ex / (np.sum(ex, axis=-1, keepdims=True) + 1e-10)
        o = np.einsum("bsht,bthd->bshd", self.p, vr)
        return self.wo.forward(o.reshape(b, s, self.d))

    def backward(self, d):
        b, s, _ = d.shape
        do = self.wo.backward(d).reshape(b, s, self.h, self.hd)
        kr, vr = np.repeat(self.k, self.g, 2), np.repeat(self.v, self.g, 2)
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


class SovereignLogic:
    def __init__(self, d):
        self.ge = [Linear(d, d * 2), Linear(d * 2, d)]
        self.gr = [Linear(d, d * 2), Linear(d * 2, d)]
        self.gt = Linear(d, 2)

    def forward(self, x):
        self.h_ge = swi(self.ge[0].forward(x))
        self.o_ge = self.ge[1].forward(self.h_ge)
        self.h_gr = swi(self.gr[0].forward(x))
        self.o_gr = self.gr[1].forward(self.h_gr)
        g = self.gt.forward(x)
        ex = np.exp(g - g.max(-1, keepdims=True))
        self.p = ex / (ex.sum(-1, keepdims=True) + 1e-10)
        return self.p[..., :1] * self.o_ge + self.p[..., 1:2] * self.o_gr

    def backward(self, d):
        dp = np.stack([(d * self.o_ge).sum(-1), (d * self.o_gr).sum(-1)], -1)
        dg = self.p * (dp - (self.p * dp).sum(-1, keepdims=True))
        dx = self.gt.backward(dg)
        dge = self.ge[1].backward(d * self.p[..., :1])
        dx += self.ge[0].backward(dswi(self.ge[0].forward(self.ge[0].x), dge))
        dgr = self.gr[1].backward(d * self.p[..., 1:2])
        dx += self.gr[0].backward(dswi(self.gr[0].forward(self.gr[0].x), dgr))
        return dx


class MoE:
    def __init__(self, d, n=4):
        self.d, self.n, self.gt = d, n, Linear(d, n)
        self.ex = [[Linear(d, d * 2), Linear(d * 2, d)] for _ in range(n)]

    def forward(self, x):
        s = x.shape
        x = x.reshape(-1, self.d)
        g = self.gt.forward(x)
        ex = np.exp(g - g.max(-1, keepdims=True))
        self.p = ex / (ex.sum(-1, keepdims=True) + 1e-10)
        self.sel = np.argmax(self.p, -1)
        out = np.zeros_like(x)
        self.c = []
        for i in range(self.n):
            m = self.sel == i
            if not np.any(m):
                self.c.append(None)
                continue
            h = swi(self.ex[i][0].forward(x[m]))
            y = self.ex[i][1].forward(h)
            out[m] = y * self.p[m, i : i + 1]
            self.c.append((m, h, y))
        return out.reshape(s)

    def backward(self, d):
        s = d.shape
        d = d.reshape(-1, self.d)
        dx, dp = np.zeros_like(d), np.zeros_like(self.p)
        for i in range(self.n):
            if self.c[i] is None:
                continue
            m, h, y = self.c[i]
            dp[m, i] = (d[m] * y).sum(-1)
            dy = d[m] * self.p[m, i : i + 1]
            dh = self.ex[i][1].backward(dy)
            dx[m] += self.ex[i][0].backward(
                dswi(self.ex[i][0].forward(self.ex[i][0].x), dh)
            )
        return (
            dx + self.gt.backward(self.p * (dp - (self.p * dp).sum(-1, True)))
        ).reshape(s)


class Block:
    def __init__(self, d):
        self.l1, self.at, self.l2, self.lg, self.l3, self.mo = (
            RMS(d),
            Attn(d),
            RMS(d),
            Logic(d),
            RMS(d),
            MoE(d),
        )

    def forward(self, x):
        x = x + self.at.forward(self.l1.forward(x))
        x = x + self.lg.forward(self.l2.forward(x))
        return x + self.mo.forward(self.l3.forward(x))

    def backward(self, d):
        d = d + self.n3.backward(self.mo.backward(d))
        d = d + self.n2.backward(self.lg.backward(d))
        return d + self.n1.backward(self.at.backward(d))


class OMEGA:
    def __init__(self, i=784, h=128, o=10, d=1):
        self.st, self.bl = Linear(i, h), [Block(h) for _ in range(d)]
        self.nm, self.hd = RMS(h), Linear(h, o)

    def forward(self, x):
        x = self.st.forward(x)[:, None, :]
        for b in self.bl:
            x = b.forward(x)
        self.f = self.nm.forward(x[:, 0, :])
        return self.hd.forward(self.f)

    def backward(self, d):
        d = self.nm.backward(self.hd.backward(d))[:, None, :]
        for b in reversed(self.bl):
            d = b.backward(d)
        self.st.backward(d[:, 0, :])

    def params(self):
        p = []

        def f(o):
            if isinstance(o, (Linear, RMS)):
                p.append(o)
            elif isinstance(o, list):
                [f(i) for i in o]
            elif hasattr(o, "__dict__"):
                [f(v) for v in o.__dict__.values()]

        f(self)
        return p


class Lion:
    def __init__(self, p, lr=1e-4, b1=0.9, b2=0.99, wd=0.01):
        self.p, self.lr, self.b1, self.b2, self.wd = p, lr, b1, b2, wd
        self.m = [np.zeros_like(getattr(x, "W", getattr(x, "g", 0))) for x in p]
        self.mb = [np.zeros_like(x.b) if hasattr(x, "b") else None for x in p]

    def step(self):
        for i, p in enumerate(self.p):
            if hasattr(p, "W"):
                for a, m in [("W", self.m), ("b", self.mb)]:
                    if m[i] is None:
                        continue
                    g, w = getattr(p, "d" + a), getattr(p, a)
                    u = np.sign(self.b1 * m[i] + (1 - self.b1) * g)
                    w -= self.lr * (u + self.wd * w if a == "W" else u)
                    m[i] = self.b2 * m[i] + (1 - self.b2) * g
                    setattr(p, a, w)
            else:
                u = np.sign(self.b1 * self.m[i] + (1 - self.b1) * p.dg)
                p.g -= self.lr * (u + self.wd * p.g)
                self.m[i] = self.b2 * self.m[i] + (1 - self.b2) * p.dg


def train():
    N, D, C = 512, 784, 10
    X, Y = np.random.randn(N, D).astype("f"), np.random.randint(0, C, N)
    m = OMEGA(D, 64, C, 1)
    opt = Lion(m.params(), 3e-4)
    for e in range(10):
        idx = np.random.permutation(N)
        tl, acc = 0, 0
        for i in range(0, N, 32):
            xb, yb = X[idx[i : i + 32]], Y[idx[i : i + 32]]
            l = m.forward(xb)
            pr = np.exp(l - l.max(1, True))
            pr /= pr.sum(1, True) + 1e-10
            tl += -np.log(pr[range(len(yb)), yb] + 1e-10).mean() * len(yb)
            acc += (pr.argmax(1) == yb).sum()
            do = pr.copy()
            do[range(len(yb)), yb] -= 1
            m.backward(do / len(yb))
            gn = np.sqrt(
                sum(
                    (getattr(p, "dW", 0) ** 2).sum()
                    + (getattr(p, "db", 0) ** 2).sum()
                    + (getattr(p, "dg", 0) ** 2).sum()
                    for p in m.params()
                )
            )
            if gn > 5:
                for p in m.params():
                    if hasattr(p, "dW"):
                        p.dW *= 5 / gn
                        p.db *= 5 / gn
                    if hasattr(p, "dg"):
                        p.dg *= 5 / gn
            opt.step()
            
        print(f"Epoch {epoch} | Loss: {total_loss/N:.4f} | Acc: {total_acc/N:.4f}")


if __name__ == "__main__":
    train()
