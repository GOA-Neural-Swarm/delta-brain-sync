import numpy as np

class P:
    def __init__(self, d): self.d, self.g = d.astype("f4"), np.zeros_like(d)

class L:
    def __init__(self, i, o):
        s = (2/(i+o))**.5
        self.w, self.b = P(np.random.randn(i, o)*s), P(np.zeros(o))
    def f(self, x): self.x = x; return x @ self.w.d + self.b.d
    def b(self, dy):
        xf, df = self.x.reshape(-1, self.x.shape[-1]), dy.reshape(-1, dy.shape[-1])
        self.w.g += xf.T @ df; self.b.g += df.sum(0)
        return dy @ self.w.d.T

class N:
    def __init__(self, d): self.g, self.e = P(np.ones(d)), 1e-6
    def f(self, x):
        self.x, self.v = x, np.mean(x**2, -1, keepdims=True)
        self.i = 1/(self.v + self.e)**.5; self.n = x * self.i
        return self.g.d * self.n
    def b(self, dy):
        self.g.g += np.sum(dy * self.n, axis=tuple(range(dy.ndim - 1)))
        dn = dy * self.g.d
        return (dn - self.n * np.mean(dn * self.n, -1, keepdims=True)) * self.i

class S:
    def f(self, x):
        x, g = np.split(x, 2, -1)
        self.s = 1/(1+np.exp(-np.clip(g, -15, 15)))
        self.sw, self.x, self.g = g * self.s, x, g
        return x * self.sw
    def b(self, dy):
        dx, dg = dy * self.sw, dy * self.x * self.s * (1 + self.g * (1 - self.s))
        return np.concatenate([dx, dg], -1)

class A:
    def __init__(self, d, h=4):
        self.d, self.h, self.hd, self.sc = d, h, d // h, (d // h)**-0.5
        self.wq, self.wk, self.wv, self.wo = L(d, d), L(d, d), L(d, d), L(d, d)
    def f(self, x):
        b, s, _ = x.shape
        self.q, self.k, self.v = [m.f(x).reshape(b, s, self.h, self.hd) for m in (self.wq, self.wk, self.wv)]
        at = np.einsum("bshd,bthd->bsht", self.q, self.k) * self.sc
        at = np.exp(at - at.max(-1, keepdims=True))
        self.p = at / (at.sum(-1, keepdims=True) + 1e-12)
        return self.wo.f(np.einsum("bsht,bthd->bshd", self.p, self.v).reshape(b, s, -1))
    def b(self, dy):
        b, s, _ = dy.shape
        dyo = self.wo.b(dy).reshape(b, s, self.h, self.hd)
        dv = np.einsum("bsht,bshd->bthd", self.p, dyo)
        da = np.einsum("bshd,bthd->bsht", dyo, self.v) * self.p * (1 - self.p) * self.sc
        dq, dk = np.einsum("bsht,bthd->bshd", da, self.k), np.einsum("bsht,bshd->bthd", da, self.q)
        return self.wq.b(dq.reshape(b, s, -1)) + self.wk.b(dk.reshape(b, s, -1)) + self.wv.b(dv.reshape(b, s, -1))

class M:
    def __init__(self, d): self.w1, self.w2, self.sw = L(d, d * 2), L(d, d), S()
    def f(self, x): return self.w2.f(self.sw.f(self.w1.f(x)))
    def b(self, dy): return self.w1.b(self.sw.b(self.w2.b(dy)))

class B:
    def __init__(self, d): self.n1, self.at, self.n2, self.ff = N(d), A(d), N(d), M(d)
    def f(self, x):
        self.x1 = x + self.at.f(self.n1.f(x))
        return self.x1 + self.ff.f(self.n2.f(self.x1))
    def b(self, dy):
        df = self.n2.b(self.ff.b(dy)) + dy
        return self.n1.b(self.at.b(df)) + df

class Model:
    def __init__(self, di, dm, do):
        self.eb, self.bl = L(di, dm), [B(dm) for _ in range(2)]
        self.fn, self.hd, self.ps = N(dm), L(dm, do), []
        self._g(self)
    def _g(self, o):
        if isinstance(o, P): self.ps.append(o)
        elif hasattr(o, "__dict__"):
            for v in o.__dict__.values():
                [self._g(i) for i in v] if isinstance(v, list) else self._g(v)
    def f(self, x):
        x = self.eb.f(x[:, None, :])
        for b in self.bl: x = b.f(x)
        return self.hd.f(self.fn.f(x[:, -1, :]))
    def b(self, dy):
        dy = self.fn.b(self.hd.b(dy))
        db = np.zeros((dy.shape[0], 1, dy.shape[1])); db[:, -1, :] = dy
        for b in reversed(self.bl): db = b.b(db)
        self.eb.b(db)

class Brain:
    def __init__(self):
        self.en, self.ho, self.re, self.ti, self.hi = 1.0, 100.0, 432.0, 1, []
        self.m, self.lr = Model(784, 128, 10), 1e-3
    def cycle(self, x, y):
        lts = self.m.f(x)
        pr = np.exp(lts - lts.max(-1, keepdims=True))
        pr /= pr.sum(-1, keepdims=True)
        loss = -np.mean(np.log(pr[np.arange(len(y)), y] + 1e-12))
        dl = pr.copy(); dl[np.arange(len(y)), y] -= 1
        self.m.b(dl / len(y))
        for p in self.m.ps: p.d -= self.lr * np.clip(p.g, -1, 1); p.g.fill(0)
        self.ho += max(0, 1 - loss); self.en += loss * 0.1; self.ti += 1; self.hi.append(loss)
        if len(self.hi) > 10:
            if np.mean(self.hi[-10:]) > 2: self.re, self.ho = self.re+5, self.ho-1
            else: self.ho += 2
        if np.random.random() < 0.1:
            for p in self.m.ps: p.d += np.random.randn(*p.d.shape) * 1e-3
        return loss
    def score(self): return (self.ho / (self.en + 1e-6)) * self.re * (1 - 1 / (self.ti + 1))

if __name__ == "__main__":
    b = Brain()
    for s in range(200):
        x, y = np.random.randn(32, 784).astype("f4"), np.random.randint(0, 10, 32)
        l = b.cycle(x, y)
        if s % 10 == 0: print(f"[{s}] L:{l:.3f}|ASI:{b.score():.1f}")