import numpy as np


class P:
    def __init__(self, d):
        self.d, self.g = d.astype("f4"), np.zeros_like(d)


class L:
    def __init__(self, i, o):
        s = (2 / (i + o)) ** 0.5
        self.w, self.b = P(np.random.randn(i, o) * s), P(np.zeros(o))

    def f(self, x):
        self.x = x
        return x @ self.w.d + self.b.d

    def b(self, dy):
        xf, df = self.x.reshape(-1, self.x.shape[-1]), dy.reshape(-1, dy.shape[-1])
        self.w.g += xf.T @ df
        self.b.g += df.sum(0)
        return dy @ self.w.d.T


class N:
    def __init__(self, d):
        self.g = P(np.ones(d))

    def f(self, x):
        self.x, self.i = x, 1 / (np.mean(x**2, -1, keepdims=True) + 1e-6) ** 0.5
        self.n = x * self.i
        return self.g.d * self.n

    def b(self, dy):
        self.g.g += np.sum(dy * self.n, axis=tuple(range(dy.ndim - 1)))
        dn = dy * self.g.d
        return (dn - self.n * np.mean(dn * self.n, -1, keepdims=True)) * self.i


class S:
    def f(self, x):
        x, g = np.split(x, 2, -1)
        self.s = 1 / (1 + np.exp(-np.clip(g, -15, 15)))
        self.x, self.g = x, g
        return x * (g * self.s)

    def b(self, dy):
        dx, dg = dy * (self.g * self.s), dy * self.x * self.s * (
            1 + self.g * (1 - self.s)
        )
        return np.concatenate([dx, dg], -1)


class A:
    def __init__(self, d, h=4):
        self.d, self.h, self.hd, self.sc = d, h, d // h, (d // h) ** -0.5
        self.wq, self.wk, self.wv, self.wo = [L(d, d) for _ in range(4)]

    def f(self, x):
        b, s, _ = x.shape
        self.q, self.k, self.v = [
            m.f(x).reshape(b, s, self.h, self.hd) for m in (self.wq, self.wk, self.wv)
        ]
        at = np.einsum("bshd,bthd->bsht", self.q, self.k) * self.sc
        at = np.exp(at - at.max(-1, keepdims=True))
        self.p = at / (at.sum(-1, keepdims=True) + 1e-9)
        return self.wo.f(np.einsum("bsht,bthd->bshd", self.p, self.v).reshape(b, s, -1))

    def b(self, dy):
        b, s, _ = dy.shape
        dyo = self.wo.b(dy).reshape(b, s, self.h, self.hd)
        dv = np.einsum("bsht,bshd->bthd", self.p, dyo)
        dp = np.einsum("bshd,bthd->bsht", dyo, self.v)
        da = self.p * (dp - (dp * self.p).sum(-1, keepdims=True)) * self.sc
        dq, dk = np.einsum("bsht,bthd->bshd", da, self.k), np.einsum(
            "bsht,bshd->bthd", da, self.q
        )
        return (
            self.wq.b(dq.reshape(b, s, -1))
            + self.wk.b(dk.reshape(b, s, -1))
            + self.wv.b(dv.reshape(b, s, -1))
        )


class B:
    def __init__(self, d):
        self.n1, self.at, self.n2, self.w1, self.w2, self.sw = (
            N(d),
            A(d),
            N(d),
            L(d, d * 2),
            L(d, d),
            S(),
        )

    def f(self, x):
        self.x1 = x + self.at.f(self.n1.f(x))
        return self.x1 + self.w2.f(self.sw.f(self.w1.f(self.n2.f(self.x1))))

    def b(self, dy):
        df = self.n2.b(self.w1.b(self.sw.b(self.w2.b(dy)))) + dy
        return self.n1.b(self.at.b(df)) + df


class Mod:
    def __init__(self, di, dm, do):
        self.eb, self.bl, self.fn, self.hd, self.ps = (
            L(di, dm),
            [B(dm) for _ in range(2)],
            N(dm),
            L(dm, do),
            [],
        )
        self._g(self)

    def _g(self, o):
        if isinstance(o, P):
            if o not in self.ps:
                self.ps.append(o)
        elif hasattr(o, "__dict__"):
            for v in o.__dict__.values():
                if v is self.ps:
                    continue
                if isinstance(v, list):
                    [self._g(i) for i in v]
                else:
                    self._g(v)

    def f(self, x):
        x = self.eb.f(x[:, None, :])
        for b in self.bl:
            x = b.f(x)
        return self.hd.f(self.fn.f(x[:, -1, :]))

    def b(self, dy):
        dy = self.fn.b(self.hd.b(dy))
        db = np.zeros((dy.shape[0], 1, dy.shape[1]))
        db[:, -1, :] = dy
        for b in reversed(self.bl):
            db = b.b(db)
        self.eb.b(db)


class Br:
    def __init__(self):
        self.en, self.ho, self.re, self.ti, self.hi, self.m, self.lr = (
            1.0,
            100.0,
            432.0,
            1,
            [],
            Mod(784, 128, 10),
            1e-3,
        )

    def cycle(self, x, y):
        lts = self.m.f(x)
        pr = np.exp(lts - lts.max(-1, keepdims=True))
        pr /= pr.sum(-1, keepdims=True)
        loss = -np.mean(np.log(pr[np.arange(len(y)), y] + 1e-9))
        dl = pr.copy()
        dl[np.arange(len(y)), y] -= 1
        self.m.b(dl / len(y))
        for p in self.m.ps:
            p.d -= self.lr * np.clip(p.g, -1, 1)
            p.g.fill(0)
        self.ho += max(0, 1 - loss)
        self.en += loss * 0.1
        self.ti += 1
        self.hi.append(loss)
        if len(self.hi) > 10:
            if np.mean(self.hi[-10:]) > 2:
                self.re += 5
                self.ho -= 1
            else:
                self.ho += 2
        if np.random.random() < 0.1:
            for p in self.m.ps:
                p.d += np.random.normal(0, 1e-3, p.d.shape)
        return loss

    def score(self):
        return (self.ho / (self.en + 1e-6)) * self.re * (1 - 1 / (self.ti + 1))


if __name__ == "__main__":
    b = Br()
    for s in range(200):
        x, y = np.random.normal(0, 1, (32, 784)).astype("f4"), np.random.randint(
            0, 10, 32
        )
        l = b.cycle(x, y)
        if s % 10 == 0:
            print(f"[{s}] L:{l:.3f}|ASI:{b.score():.1f}")