import numpy as np


class T:
    def __init__(self, d):
        self.data = d.astype("f4")
        self.grad = np.zeros_like(self.data)


class M:
    def params(self):
        p = []
        for v in self.__dict__.values():
            if isinstance(v, T):
                p.append(v)
            elif isinstance(v, M):
                p.extend(v.params())
            elif isinstance(v, list):
                for i in v:
                    p.extend(
                        i.params()
                        if isinstance(i, M)
                        else ([i] if isinstance(i, T) else [])
                    )
        return p


class L(M):
    def __init__(self, i, o, b=True):
        self.w = T(np.random.randn(i, o) * (2 / i) ** 0.5)
        self.b = T(np.zeros(o)) if b else None

    def f(self, x):
        self.x = x
        return x @ self.w.data + (self.b.data if self.b else 0)

    def b(self, dy):
        xf = self.x.reshape(-1, self.x.shape[-1])
        df = dy.reshape(-1, dy.shape[-1])
        self.w.grad += xf.T @ df
        if self.b:
            self.b.grad += df.sum(0)
        return (dy @ self.w.data.T).reshape(self.x.shape)


class RN(M):
    def __init__(self, d, e=1e-6):
        self.g, self.e = T(np.ones(d)), e

    def f(self, x):
        self.x, self.v = x, np.mean(x**2, -1, keepdims=True)
        self.i = (self.v + self.e) ** -0.5
        self.nx = x * self.i
        return self.g.data * self.nx

    def b(self, dy):
        dg = dy * self.g.data
        self.g.grad += np.sum(dy * self.nx, axis=tuple(range(dy.ndim - 1)))
        return (dg - self.nx * np.mean(dg * self.nx, -1, keepdims=True)) * self.i


class SG(M):
    def f(self, x):
        self.x = x
        self.g, self.v = np.split(x, 2, -1)
        self.sig = 1 / (1 + np.exp(-np.clip(self.g, -12, 12)))
        return (self.g * self.sig) * self.v

    def b(self, dy):
        ds = dy * self.v
        dv = dy * self.swish
        dg = ds * self.sig * (1 + self.g * (1 - self.sig))
        return np.concatenate([dg, dv], -1)


class GQA(M):
    def __init__(self, d, h=8, g=2):
        self.d, self.h, self.g, self.hd = d, h, g, d // h
        self.wq, self.wk, self.wv, self.wo = (
            L(d, d),
            L(d, (h // g) * self.hd),
            L(d, (h // g) * self.hd),
            L(d, d),
        )
        self.sc = self.hd**-0.5
        f = 10000 ** -(np.arange(0, self.hd, 2) / self.hd)
        t = np.arange(2048)[:, None] * f
        self.cos, self.sin = np.cos(t), np.sin(t)

    def _rp(self, x, v=False):
        s = x.shape[1]
        c, sn = self.cos[:s, None, :], self.sin[:s, None, :] * (-1 if v else 1)
        r, i = x[..., ::2], x[..., 1::2]
        o = np.empty_like(x)
        o[..., ::2], o[..., 1::2] = r * c - i * sn, r * sn + i * c
        return o

    def f(self, x):
        b, s, _ = x.shape
        q, k, v = (
            self.wq.f(x).reshape(b, s, self.h, self.hd),
            self.wk.f(x).reshape(b, s, self.h // self.g, self.hd),
            self.wv.f(x).reshape(b, s, self.h // self.g, self.hd),
        )
        self.qr, self.kr, self.v_o = self._rp(q), self._rp(k), v
        at = (
            np.einsum("bshd,bthd->bsht", self.qr, np.repeat(self.kr, self.g, 2))
            * self.sc
        )
        self.p = (e := np.exp(at - at.max(-1, keepdims=True))) / (
            e.sum(-1, keepdims=True) + 1e-12
        )
        return self.wo.f(
            np.einsum("bsht,bthd->bshd", self.p, np.repeat(v, self.g, 2)).reshape(
                b, s, -1
            )
        )

    def b(self, dy):
        b, s = dy.shape[:2]
        do = self.w_o.b(dy).reshape(b, s, self.h, self.hd)
        dp = np.einsum("bshd,bthd->bsht", do, self.kv)
        da = self.p * (dp - (self.p * dp).sum(-1, keepdims=True)) * self.sc
        dq = self._rp(
            np.einsum("bsht,bthd->bshd", da, np.repeat(self.kr, self.g, 2)), True
        )
        dk = self._rp(
            np.einsum("bsht,bshd->bthd", da, self.qr)
            .reshape(b, s, -1, self.g, self.hd)
            .sum(3),
            True,
        )
        dv = (
            np.einsum("bsht,bshd->bthd", self.p, dyw)
            .reshape(b, s, -1, self.g, self.hd)
            .sum(3)
        )
        return (
            self.wq.b(dq.reshape(b, s, -1))
            + self.wk.b(dk.reshape(b, s, -1))
            + self.wv.b(dv.reshape(b, s, -1))
        )


class MoE(M):
    def __init__(self, d, n=4, k=2):
        self.d, self.n, self.k, self.gate = d, n, k, L(d, n)
        self.ex = [[L(d, d * 2), SG(), L(d, d)] for _ in range(n)]

    def f(self, x):
        self.sh, xf = x.shape, x.reshape(-1, self.d)
        p = (
            e := np.exp(l := self.gate.f(xf) - self.gate.f(xf).max(-1, keepdims=True))
        ) / (e.sum(-1, keepdims=True) + 1e-12)
        self.idx = np.argsort(p, -1)[:, -self.k :]
        self.w = np.take_along_axis(p, self.idx, -1)
        self.w /= self.w.sum(-1, keepdims=True) + 1e-12
        out, self.cache = np.zeros_like(xf), []
        for i in range(self.n):
            m = np.any(self.idx == i, -1)
            if not np.any(m):
                self.c.append(None)
                continue
            ps = np.where(self.idx[m] == i)[1]
            h = self.experts[i][2].f(self.experts[i][1].f(self.experts[i][0].f(xf[m])))
            out[m] += h * self.w[m, ps][:, None]
            self.cache.append((m, ps, h))
        return out.reshape(self.sh)

    def b(self, dy):
        dyf, dx, dg = (
            dy.reshape(-1, self.d),
            np.zeros((dy.size // self.d, self.d)),
            np.zeros((dy.size // self.d, self.n)),
        )
        for i in range(self.n):
            if self.cache[i] is not None:
                m, ps, h = self.cache[i]
                dg[m, i] = (dyf[m] * h).sum(-1)
                dx[m] += self.ex[i][0].b(
                    self.ex[i][1].b(self.ex[i][2].b(dyf[m] * self.w[m, ps][:, None]))
                )
        return (dx + self.gate.b(dg - dg.mean(-1, keepdims=True))).reshape(self.sh)


class DSL(M):
    def __init__(self, d):
        self.p1, self.p2, self.fuz = (
            [L(d, d * 2), SG(), L(d, d)],
            [L(d, d * 2), SG(), L(d, d)],
            L(d * 2, d),
        )

    def f(self, x):
        self.o1 = self.p1[2].f(self.p1[1].f(self.p1[0].f(x)))
        self.o2 = self.p2[2].f(self.p2[1].f(self.p2[0].f(x)))
        return self.fuz.f(np.concatenate([self.o1, self.o2], -1))

    def b(self, dy):
        d1, d2 = np.split(self.fuz.b(dy), 2, -1)
        return self.p1[0].b(self.p1[1].b(self.p1[2].b(d1))) + self.p2[0].b(
            self.p2[1].b(self.p2[2].b(d2))
        )


class Block(M):
    def __init__(self, d):
        self.n1, self.at, self.n2, self.ds, self.n3, self.mo = (
            RN(d),
            GQA(d),
            RN(d),
            DSL(d),
            RN(d),
            RE(d),
        )

    def f(self, x):
        x = x + self.at.f(self.n1.f(x))
        x = x + self.ds.f(self.n2.f(x))
        return x + self.mo.f(self.n3.f(x))

    def b(self, dy):
        dy = dy + self.ff.b(self.n2.b(dy))
        return dy + self.at.b(self.n1.b(dy))


class ASI(M):
    def __init__(self, di, dm, do, dp=3):
        self.em, self.bl = L(di, dm), [B(dm) for _ in range(dp)]
        self.nm, self.hd = RN(dm), L(dm, do)

    def f(self, x):
        x = self.em.f(x[:, None] if x.ndim == 2 else x)
        for b in self.bl:
            x = b.f(x)
        return self.hd.f(self.nm.f(x[:, -1]))

    def b(self, dy):
        dy = self.norm.b(self.head.b(dy))
        db = np.zeros((dy.shape[0], self.emb.x.shape[1], dy.shape[1]))
        db[:, -1] = dy
        for b in reversed(self.bl):
            db = b.b(db)
        self.em.b(db)


class Adam:
    def __init__(self, p, lr=1e-3, b1=0.9, b2=0.999, wd=0.01):
        self.p, self.lr, self.b1, self.b2, self.wd, self.t = p, lr, b1, b2, wd, 0
        self.m, self.v = [np.zeros_like(i.data) for i in p], [
            np.zeros_like(i.data) for i in p
        ]

    def step(self):
        self.t += 1
        a = self.lr * (np.sqrt(1 - self.b2**self.t) / (1 - self.b1**self.t))
        for i, pt in enumerate(self.p):
            g = np.clip(pt.grad, -10, 10)
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g**2)
            pt.data -= a * (self.m[i] / (np.sqrt(self.v[i]) + 1e-8) + self.wd * pt.data)
            pt.grad.fill(0)


if __name__ == "__main__":
    N, D, C, BS, E = 1024, 784, 10, 64, 50
    X, Y = np.random.randn(N, D), np.random.randint(0, C, N)
    m = ASI(D, 128, C, 2)
    opt = Adam(m.params(), 2e-3)
    for e in range(E):
        idx = np.random.permutation(N)
        losses, accs = [], []
        for i in range(0, N, BS):
            xb, yb = X[idx[i : i + BS]], Y[idx[i : i + BS]]
            lg = m.f(xb)
            pr = (ex := np.exp(lg - lg.max(-1, 1, keepdims=True))) / (
                ex.sum(-1, keepdims=True) + 1e-12
            )
            ls.append(-np.mean(np.log(pr[range(len(yb)), yb] + 1e-12)))
            ac.append(np.mean(pr.argmax(-1) == yb))
            dl = pr.copy()
            dl[range(len(yb)), yb] -= 1
            m.b(dl / len(yb))
            opt.step()
        if (e + 1) % 5 == 0:
            print(f"E {e+1:02} | L: {np.mean(ls):.4f} | A: {np.mean(ac):.4f}")
