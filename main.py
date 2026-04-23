import numpy as np

class P:
    def __init__(self, d):
        self.d = d.astype("f4")
        self.g = np.zeros_like(d)

class L:
    def __init__(self, i, o):
        s = (2/(i+o))**.5
        self.w, self.b = P(np.random.randn(i,o)*s), P(np.zeros(o))
    def f(self, x):
        self.x = x
        return x @ self.w.d + self.b.d
    def b(self, dy):
        xf = self.x.reshape(-1, self.x.shape[-1])
        df = dy.reshape(-1, dy.shape[-1])
        self.w.g += xf.T @ df
        self.b.g += df.sum(0)
        return dy @ self.w.d.T

class N:
    def __init__(self, d):
        self.g, self.e = P(np.ones(d)), 1e-6
    def f(self, x):
        self.x = x
        self.v = np.mean(x**2, -1, keepdims=1)
        self.i = 1/(self.v + self.e)**.5
        self.n = x * self.i
        return self.g.d * self.n
    def b(self, dy):
        self.g.g += np.sum(dy*self.n, axis=tuple(range(dy.ndim-1)))
        dn = dy * self.g.d
        return (dn - self.n * np.mean(dn*self.n, -1, keepdims=1)) * self.i

class S:
    def f(self, x):
        x, g = np.split(x, 2, -1)
        self.s = 1/(1 + np.exp(-np.clip(g, -15, 15)))
        self.sw, self.x, self.g = g * self.s, x, g
        return x * self.sw
    def b(self, dy):
        dx = dy * self.sw
        dg = dy * self.x * self.s * (1 + self.g * (1 - self.s))
        return np.concatenate([dx, dg], -1)

class A:
    def __init__(self, d, h=8, g=2):
        self.d, self.h, self.g, self.hd = d, h, g, d//h
        self.wq, self.wk = L(d, d), L(d, (h//g)*self.hd)
        self.wv, self.wo = L(d, (h//g)*self.hd), L(d, d)
        self.sc = self.hd**-.5
    def _r(self, t, c, s):
        r, i = t[..., ::2], t[..., 1::2]
        return np.stack([r*c-i*s, r*s+i*c], -1).reshape(t.shape)
    def f(self, x):
        b, s, _ = x.shape
        q = self.wq.f(x).reshape(b, s, self.h, self.hd)
        k = self.wk.f(x).reshape(b, s, self.h//self.g, self.hd)
        v = self.wv.f(x).reshape(b, s, self.h//self.g, self.hd)
        a = np.arange(s)[:, None] * (10000**-(np.arange(0, self.hd, 2)/self.hd))
        co, si = np.cos(a)[:, None, :], np.sin(a)[:, None, :]
        self.qr, self.kr, self.vr = self._r(q, co, si), self._r(k, co, si), v
        ke, ve = np.repeat(self.kr, self.g, 2), np.repeat(v, self.g, 2)
        at = np.einsum("bshd,bthd->bsht", self.qr, ke) * self.sc
        at = np.exp(at - np.max(at, -1, keepdims=1))
        self.p = at / (at.sum(-1, keepdims=1) + 1e-12)
        return self.wo.f(np.einsum("bsht,bthd->bshd", self.p, ve).reshape(b, s, -1))
    def b(self, dy):
        b, s = dy.shape[:2]
        dw = self.wo.b(dy).reshape(b, s, self.h, self.hd)
        ke, ve = np.repeat(self.kr, self.g, 2), np.repeat(self.vr, self.g, 2)
        dp = np.einsum("bshd,bthd->bsht", dw, ve)
        da = self.p * (dp - np.sum(self.p*dp, -1, keepdims=1)) * self.sc
        dq = np.einsum("bsht,bthd->bshd", da, ke)
        dk = np.einsum("bsht,bshd->bthd", da, self.qr).reshape(b, s, self.h//self.g, self.g, self.hd).sum(3)
        dv = np.einsum("bsht,bshd->bthd", self.p, dw).reshape(b, s, self.h//self.g, self.g, self.hd).sum(3)
        return self.wq.b(dq.reshape(b, s, -1)) + self.wk.b(dk.reshape(b, s, -1)) + self.wv.b(dv.reshape(b, s, -1))

class M:
    def __init__(self, d, n=4, k=2):
        self.d, self.n, self.k = d, n, k
        self.gt = L(d, n)
        self.w1 = [P(np.random.randn(d, d*2)*(2/(d*3))**.5) for _ in range(n)]
        self.w2 = [P(np.random.randn(d*2, d)*d**-.5) for _ in range(n)]
        self.sw = [S() for _ in range(n)]
    def f(self, x):
        sh, x = x.shape, x.reshape(-1, self.d)
        g = self.gt.f(x)
        p = np.exp(g - g.max(-1, keepdims=1))
        p /= p.sum(-1, keepdims=1)
        self.ix = np.argsort(p, -1)[:, -self.k:]
        self.wt = np.take_along_axis(p, self.ix, -1)
        self.wt /= self.wt.sum(-1, keepdims=1) + 1e-12
        o, self.ch = np.zeros_like(x), []
        for i in range(self.n):
            m = np.any(self.ix == i, -1)
            if not np.any(m): self.ch.append(None); continue
            ps = np.where(self.ix[m] == i)[1][:, None]
            xi = x[m]
            h1 = xi @ self.w1[i].d
            at = self.sw[i].f(h1)
            wi = self.wt[m, ps[:, 0]][:, None]
            o[m] += (at @ self.w2[i].d) * wi
            self.ch.append((m, xi, at, wi))
        return o.reshape(sh)
    def b(self, dy):
        df = dy.reshape(-1, self.d)
        dx, dg = np.zeros_like(df), np.zeros((df.shape[0], self.n))
        for i in range(self.n):
            if self.ch[i] is None: continue
            m, xi, at, wi = self.ch[i]
            di = df[m] * wi
            dg[m, i] = np.sum(df[m] * (at @ self.w2[i].d), -1)
            self.w2[i].g += at.T @ di
            da = self.sw[i].b(di @ self.w2[i].d.T)
            self.w1[i].g += xi.T @ da
            dx[m] += da @ self.w1[i].d.T
        return (dx + self.gt.b(dg - np.mean(dg, -1, keepdims=1))).reshape(dy.shape)

class B:
    def __init__(self, d):
        self.n1, self.at, self.n2, self.me = N(d), A(d), N(d), M(d)
        self.p = P(np.array([.5]))
    def f(self, x):
        self.x1 = x + self.at.f(self.n1.f(x))
        self.m = self.me.f(self.n2.f(self.x1))
        return self.x1 + (self.p.d * self.m + (1 - self.p.d) * self.x1)
    def b(self, dy):
        self.p.g += np.sum(dy * (self.m - self.x1))
        dx1 = dy * (1 - self.p.d) + self.n2.b(self.me.b(dy * self.p.d))
        return dx1 + self.n1.b(self.at.b(dx1))

class Model:
    def __init__(self, di, dm, do, dp=2):
        self.eb, self.bl = L(di, dm), [B(dm) for _ in range(dp)]
        self.fn, self.hd = N(dm), L(dm, do)
        self.ps = []
        self._g(self)
    def _g(self, o):
        if isinstance(o, P): self.ps.append(o)
        elif hasattr(o, "__dict__"):
            for v in o.__dict__.values():
                if isinstance(v, list): [self._g(i) for i in v]
                else: self._g(v)
    def f(self, x):
        x = self.eb.f(x[:, None, :] if x.ndim == 2 else x)
        for b in self.bl: x = b.f(x)
        return self.hd.f(self.fn.f(x[:, -1, :]))
    def b(self, dy):
        dy = self.fn.b(self.hd.b(dy))
        db = np.zeros((dy.shape[0], self.eb.x.shape[1], dy.shape[1]), "f4")
        db[:, -1, :] = dy
        for b in reversed(self.bl): db = b.b(db)
        self.eb.b(db)

class Opt:
    def __init__(self, ps, lr=1e-3, wd=.01):
        self.ps, self.lr, self.wd, self.t = ps, lr, wd, 0
        self.m = [np.zeros_like(p.d) for p in ps]
        self.v = [np.zeros_like(p.d) for p in ps]
    def step(self):
        self.t += 1
        lt = self.lr * ((1 - .999**self.t)**.5 / (1 - .9**self.t))
        for i, p in enumerate(self.ps):
            g = np.clip(p.g, -1, 1)
            self.m[i] = .9 * self.m[i] + .1 * g
            self.v[i] = .999 * self.v[i] + .001 * (g**2)
            p.d -= lt * (self.m[i] / (np.sqrt(self.v[i]) + 1e-8) + self.wd * p.d)
            p.g.fill(0)

def train():
    NS, D, C, BS, E = 1024, 784, 10, 64, 50
    X, Y = np.random.randn(NS, D).astype("f4"), np.random.randint(0, C, NS)
    m = Model(D, 128, C)
    opt = Opt(m.ps, 3e-3)
    for e in range(E):
        ix, ls, ac = np.random.permutation(NS), [], []
        for i in range(0, NS, BS):
            xb, yb = X[ix[i:i+BS]], Y[ix[i:i+BS]]
            lg = m.f(xb)
            pr = np.exp(lg - lg.max(-1, keepdims=1))
            pr /= pr.sum(-1, keepdims=1)
            ls.append(-np.mean(np.log(pr[np.arange(len(yb)), yb] + 1e-12)))
            ac.append(np.mean(np.argmax(pr, -1) == yb))
            dl = pr.copy(); dl[np.arange(len(yb)), yb] -= 1
            m.b(dl / len(yb)); opt.step()
        if (e + 1) % 5 == 0: print(f"E {e+1} | L {np.mean(ls):.4f} | A {np.mean(ac):.4f}")

if __name__ == "__main__":
    train()