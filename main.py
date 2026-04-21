import numpy as np


def sm(x, a=-1):
    e = np.exp(x - x.max(a, keepdims=1))
    return e / (e.sum(a, keepdims=1) + 1e-9)


class T:
    def __init__(self, d):
        self.data, self.grad = d.astype("f4"), np.zeros_like(d, "f4")


class M:
    def params(self):
        p = []
        for v in self.__dict__.values():
            if isinstance(v, T):
                p += [v]
            elif isinstance(v, M):
                p += v.params()
            elif isinstance(v, list):
                for i in v:
                    if isinstance(i, M):
                        p += i.params()
                    elif isinstance(i, list):
                        p += sum([j.params() for j in i if isinstance(j, M)], [])
        return p


class L(M):
    def __init__(self, i, o, b=0):
        self.w = T(np.random.randn(i, o) * (2 / i) ** 0.5)
        self.b_ = T(np.zeros(o)) if b else None

    def f(self, x):
        self.x = x
        return x @ self.w.data + (self.b_.data if self.b_ else 0)

    def b(self, dy):
        xf, df = self.x.reshape(-1, self.x.shape[-1]), dy.reshape(-1, dy.shape[-1])
        self.w.grad += xf.T @ df
        if self.b_:
            self.b_.grad += df.sum(0)
        return (dy @ self.w.data.T).reshape(self.x.shape)


class N(M):
    def __init__(self, d, e=1e-6):
        self.g, self.e = T(np.ones(d)), e

    def f(self, x):
        self.x, self.r = x, 1 / np.sqrt((x * x).mean(-1, keepdims=1) + self.e)
        return self.g.data * (x * self.r)

    def b(self, dy):
        xn = self.x * self.r
        self.g.grad += (dy * xn).sum(tuple(range(dy.ndim - 1)))
        dn = dy * self.g.data
        return self.r * (dn - xn * (dn * xn).mean(-1, keepdims=1))


class G(M):
    def f(self, x):
        self.x1, self.x2 = np.split(x, 2, -1)
        self.s = 1 / (1 + np.exp(-np.clip(self.x1, -10, 10)))
        return (self.x1 * self.s) * self.x2

    def b(self, dy):
        ds, dx2 = dy * self.x2, dy * (self.x1 * self.s)
        dx1 = ds * (self.s * (1 + self.x1 * (1 - self.s)))
        return np.concatenate([dx1, dx2], -1)


class R(M):
    def __init__(self, d, m=2048):
        f = np.outer(np.arange(m), 1 / (10000 ** (np.arange(0, d, 2) / d)))
        self.c, self.s = np.cos(f), np.sin(f)

    def apply(self, x, v=0):
        s = x.shape[1]
        c, sn = self.c[:s, None, :], self.s[:s, None, :]
        r, i = x[..., ::2], x[..., 1::2]
        o = np.empty_like(x)
        if not v:
            o[..., ::2], o[..., 1::2] = r * c - i * sn, r * sn + i * c
        else:
            o[..., ::2], o[..., 1::2] = r * c + i * sn, i * c - r * sn
        return o


class A(M):
    def __init__(self, d, h=8, r=None):
        self.h, self.hd, self.r = h, d // h, r
        self.wq, self.wk, self.wv, self.wo = [L(d, d) for _ in "1234"]

    def f(self, x):
        b, s, _ = x.shape
        q, k, v = [
            getattr(self, f"w{i}").f(x).reshape(b, s, self.h, self.hd) for i in "qkv"
        ]
        if self.r:
            q, k = self.r.apply(q), self.r.apply(k)
        self.q, self.k, self.v, self.p = (
            q,
            k,
            v,
            sm(np.einsum("bshd,bthd->bsht", q, k) * (self.hd**-0.5)),
        )
        return self.wo.f(np.einsum("bsht,bthd->bshd", self.p, v).reshape(b, s, -1))

    def b(self, dy):
        b, s = dy.shape[:2]
        do = self.wo.b(dy).reshape(b, s, self.h, self.hd)
        dp = np.einsum("bshd,bthd->bsht", do, self.v)
        ds = self.p * (dp - (self.p * dp).sum(-1, keepdims=1)) * (self.hd**-0.5)
        dq, dk, dv = (
            np.einsum("bsht,bthd->bshd", ds, self.k),
            np.einsum("bsht,bshd->bthd", ds, self.q),
            np.einsum("bsht,bshd->bthd", self.p, do),
        )
        if self.r:
            dq, dk = self.r.apply(dq, 1), self.r.apply(dk, 1)
        return (
            self.wq.b(dq.reshape(b, s, -1))
            + self.wk.b(dk.reshape(b, s, -1))
            + self.wv.b(dv.reshape(b, s, -1))
        )


class E(M):
    def __init__(self, d, n=4, k=2):
        self.d, self.n, self.k, self.gate = d, n, k, L(d, n)
        self.exp = [[L(d, d * 2), G(), L(d * 2, d)] for _ in range(n)]

    def f(self, x):
        self.sh = x.shape
        xf = x.reshape(-1, self.d)
        self.p = sm(self.gate.f(xf))
        self.idx = np.argsort(self.p, -1)[:, -self.k :]
        self.w = np.take_along_axis(self.p, self.idx, -1)
        self.w /= self.w.sum(-1, keepdims=1) + 1e-9
        out, self.cache = np.zeros_like(xf), []
        for i in range(self.n):
            m = np.any(self.idx == i, -1)
            if not np.any(m):
                self.cache.append(None)
                continue
            pos = np.where(self.idx[m] == i)[1]
            h1 = self.exp[i][0].f(xf[m])
            h2 = self.exp[i][1].f(h1)
            h3 = self.exp[i][2].f(h2)
            out[m] += h3 * self.w[m, pos][:, None]
            self.cache.append((m, pos, h1, h2, h3))
        return out.reshape(self.sh)

    def b(self, dy):
        dyf, xf = dy.reshape(-1, self.d), self.gate.x
        dx, dg = np.zeros_like(xf), np.zeros_like(self.p)
        for i in range(self.n):
            if self.cache[i] is None:
                continue
            m, pos, h1, h2, h3 = self.cache[i]
            dg[m, i] = (dyf[m] * h3).sum(-1)
            dh2 = self.exp[i][2].b(dyf[m] * self.w[m, pos][:, None])
            dx[m] += self.exp[i][0].b(self.exp[i][1].b(dh2))
        return (
            dx + self.gate.b(self.p * (dg - (self.p * dg).sum(-1, keepdims=1)))
        ).reshape(self.sh)


class B(M):
    def __init__(self, d, r):
        self.n1, self.n2, self.at, self.mo, self.fs = (
            N(d),
            N(d),
            A(d, r=r),
            E(d),
            L(d, 2),
        )

    def f(self, x):
        self.x, self.ao, self.moo = x, self.at.f(self.n1.f(x)), self.mo.f(self.n2.f(x))
        self.g = sm(self.fs.f(x.mean(1)))
        return x + self.g[:, :1, None] * self.ao + self.g[:, 1:, None] * self.moo

    def b(self, dy):
        dg = np.stack([(dy * self.ao).sum((1, 2)), (dy * self.moo).sum((1, 2))], 1)
        df = self.fs.b(self.g * (dg - (self.g * dg).sum(-1, keepdims=1)))
        dx = (
            dy
            + self.n1.b(self.at.b(dy * self.g[:, :1, None]))
            + self.n2.b(self.mo.b(dy * self.g[:, 1:, None]))
        )
        return dx + df[:, None, :] / self.x.shape[1]


class Net(M):
    def __init__(self, di, dm, do, d=2):
        self.emb, self.rp = L(di, dm), R(dm // 8)
        self.blks = [B(dm, self.rp) for _ in range(d)]
        self.norm, self.head = N(dm), L(dm, do)

    def f(self, x):
        x = self.emb.f(x[:, None] if x.ndim == 2 else x)
        for b in self.blks:
            x = b.f(x)
        return self.head.f(self.norm.f(x[:, -1]))

    def b(self, dy):
        dy = self.norm.b(self.head.b(dy))
        db = np.zeros((dy.shape[0], self.emb.x.shape[1], dy.shape[1]), "f4")
        db[:, -1] = dy
        for b in reversed(self.blks):
            db = b.b(db)
        return self.emb.b(db)


class Opt:
    def __init__(self, p, lr=1e-3, b1=0.9, b2=0.999, wd=0.01):
        self.p, self.lr, self.b1, self.b2, self.wd, self.t = p, lr, b1, b2, wd, 0
        self.m = [np.zeros_like(x.data) for x in p]
        self.v = [np.zeros_like(x.data) for x in p]

    def step(self):
        self.t += 1
        a = self.lr * ((1 - self.b2**self.t) ** 0.5 / (1 - self.b1**self.t))
        for i, p in enumerate(self.p):
            g = np.clip(p.grad, -1, 1)
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            p.data -= a * (self.m[i] / (np.sqrt(self.v[i]) + 1e-8) + self.wd * p.data)
            p.grad.fill(0)


def train():
    N_, D, C, BS, EP = 512, 784, 10, 64, 20
    X, Y = np.random.randn(N_, D).astype("f4"), np.random.randint(0, C, N_)
    m = Net(D, 128, C)
    opt = Opt(m.params(), 1e-3, wd=0.1)
    for e in range(EP):
        idx = np.random.permutation(N_)
        mtr = []
        for i in range(0, N_, BS):
            xb, yb = X[idx[i : i + BS]], Y[idx[i : i + BS]]
            p = sm(m.f(xb))
            mtr += [
                [
                    -np.mean(np.log(p[range(len(yb)), yb] + 1e-9)),
                    np.mean(p.argmax(1) == yb),
                ]
            ]
            dl = p.copy()
            dl[range(len(yb)), yb] -= 1
            m.b(dl / len(yb))
            opt.step()
        if (e + 1) % 5 == 0:
            l, a = np.mean(mtr, 0)
            print(f"E {e+1:03} | L: {l:.4f} | A: {a:.4f}")


if __name__ == "__main__":
    train()
