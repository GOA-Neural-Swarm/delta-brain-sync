import numpy as np

class T:
    def __init__(self, d):
        self.d = d.astype("f4")
        self.g = np.zeros_like(d, dtype="f4")

class L:
    def __init__(self, i, o):
        s = (2. / (i + o))**0.5
        self.w, self.b = T(np.random.normal(0, s, (i, o))), T(np.zeros(o))
    def f(self, x):
        self.x = x
        return x @ self.w.d + self.b.d
    def b(self, dy):
        dx = dy.reshape(-1, dy.shape[-1])
        self.w.g += self.x.reshape(-1, self.x.shape[-1]).T @ dx
        self.b.g += dx.sum(0)
        return dy @ self.w.d.T

class RN:
    def __init__(self, d, e=1e-6):
        self.g, self.e = T(np.ones(d)), e
    def f(self, x):
        self.x, self.v = x, np.mean(x**2, -1, keepdims=True)
        self.isd = 1. / (self.v + self.e)**0.5
        self.nx = x * self.isd
        return self.g.d * self.nx
    def b(self, dy):
        self.g.g += np.sum(dy * self.nx, axis=tuple(range(dy.ndim - 1)))
        dn = dy * self.g.d
        return (dn - self.nx * np.mean(dn * self.nx, -1, keepdims=True)) * self.isd

class SG:
    def f(self, x):
        x, g = np.split(x, 2, -1)
        self.s = 1. / (1. + np.exp(-np.clip(g, -15, 15)))
        self.sw, self.x, self.g = g * self.s, x, g
        return x * self.sw
    def b(self, dy):
        dx = dy * self.sw
        dg = dy * self.x * self.s * (1. + self.g * (1. - self.s))
        return np.concatenate([dx, dg], -1)

class GQA:
    def __init__(self, d, h=8, g=2):
        self.d, self.h, self.g, self.hd = d, h, g, d // h
        self.wq, self.wk, self.wv, self.wo = L(d, d), L(d, (h // g) * self.hd), L(d, (h // g) * self.hd), L(d, d)
        self.sc = self.hd**-0.5
    def rope(self, t, co, si):
        r, i = t[..., ::2], t[..., 1::2]
        res = np.empty_like(t)
        res[..., ::2], res[..., 1::2] = r * co - i * si, r * si + i * co
        return res
    def f(self, x):
        b, s, _ = x.shape
        q = self.wq.f(x).reshape(b, s, self.h, self.hd)
        k = self.wk.f(x).reshape(b, s, self.h // self.g, self.hd)
        v = self.wv.f(x).reshape(b, s, self.h // self.g, self.hd)
        p, f = np.arange(s)[:, None], 10000**-(np.arange(0, self.hd, 2) / self.hd)
        a = p * f
        co, si = np.cos(a)[:, None, :], np.sin(a)[:, None, :]
        self.qr, self.kr, self.vr = self.rope(q, co, si), self.rope(k, co, si), v
        kr_r = np.repeat(self.kr, self.g, 2)
        at = np.einsum("bshd,bthd->bsht", self.qr, kr_r) * self.sc
        at = np.exp(at - np.max(at, -1, keepdims=True))
        self.p = at / (at.sum(-1, keepdims=True) + 1e-12)
        out = np.einsum("bsht,bthd->bshd", self.p, np.repeat(v, self.g, 2)).reshape(b, s, -1)
        return self.wo.f(out)
    def b(self, dy):
        b, s = dy.shape[:2]
        dw = self.wo.b(dy).reshape(b, s, self.h, self.hd)
        vr_r = np.repeat(self.vr, self.g, 2)
        dp = np.einsum("bshd,bthd->bsht", dw, vr_r)
        da = self.p * (dp - np.sum(self.p * dp, -1, keepdims=True)) * self.sc
        dqr = np.einsum("bsht,bthd->bshd", da, np.repeat(self.kr, self.g, 2))
        dkr = np.einsum("bsht,bshd->bthd", da, self.qr).reshape(b, s, self.h // self.g, self.g, self.hd).sum(3)
        dv = np.einsum("bsht,bshd->bthd", self.p, dw).reshape(b, s, self.h // self.g, self.g, self.hd).sum(3)
        return self.wq.b(dqr.reshape(b, s, -1)) + self.wk.b(dkr.reshape(b, s, -1)) + self.wv.b(dv.reshape(b, s, -1))

class MOE:
    def __init__(self, d, n=4, k=2):
        self.d, self.n, self.k = d, n, k
        self.gate = L(d, n)
        self.w1 = [T(np.random.normal(0, (2/(d+d*2))**.5, (d, d*2))) for _ in range(n)]
        self.w2 = [T(np.random.normal(0, (1/d)**.5, (d*2, d))) for _ in range(n)]
        self.swi = [SG() for _ in range(n)]
    def f(self, x):
        sh = x.shape
        x = x.reshape(-1, self.d)
        lg = self.gate.f(x)
        pr = np.exp(lg - lg.max(-1, keepdims=True))
        pr /= pr.sum(-1, keepdims=True)
        self.idx = np.argsort(pr, -1)[:, -self.k:]
        self.w = np.take_along_axis(pr, self.idx, -1)
        self.w /= (self.w.sum(-1, keepdims=True) + 1e-12)
        out, self.c = np.zeros_like(x), []
        for i in range(self.n):
            m = np.any(self.idx == i, -1)
            if not np.any(m): self.c.append(None); continue
            ix = np.where(self.idx[m] == i)[1]
            xi, wi = x[m], self.w[m, ix][:, None]
            h1 = xi @ self.w1[i].d
            act = self.swi[i].f(h1)
            out[m] += (act @ self.w2[i].d) * wi
            self.c.append((m, xi, act, wi, ix))
        return out.reshape(sh)
    def b(self, dy):
        df = dy.reshape(-1, self.d)
        dx, dg = np.zeros_like(df), np.zeros((df.shape[0], self.n))
        for i in range(self.n):
            if self.c[i] is None: continue
            m, xi, act, w, ix = self.c[i]
            dyi = df[m] * w
            dg[m, i] = np.sum(df[m] * (act @ self.w2[i].d), -1)
            self.w2[i].g += act.T @ dyi
            dh = self.swi[i].b(dyi @ self.w2[i].d.T)
            self.w1[i].g += xi.T @ dh
            dx[m] += dh @ self.w1[i].d.T
        return (dx + self.gate.b(dg - dg.mean(-1, keepdims=True))).reshape(dy.shape)

class SB:
    def __init__(self, d):
        self.n1, self.at, self.n2, self.moe, self.pg = RN(d), GQA(d), RN(d), MOE(d), T(np.array([0.5]))
    def f(self, x):
        x = x + self.at.f(self.n1.f(x))
        m = self.moe.f(self.n2.f(x))
        return x + (self.pg.d * m + (1 - self.pg.d) * x)
    def b(self, dy):
        dm = dy * self.pg.d
        dx_m = self.n2.b(self.moe.b(dm))
        dy = dy + dx_m
        return dy + self.n1.b(self.at.b(dy))

class OMEGA:
    def __init__(self, di, dm, do, depth=2):
        self.emb, self.blks, self.fn, self.hd = L(di, dm), [SB(dm) for _ in range(depth)], RN(dm), L(dm, do)
        self.ps = self._gp()
    def _gp(self):
        p = []
        def w(o):
            if isinstance(o, T): p.append(o)
            elif isinstance(o, list): [w(i) for i in o]
            elif hasattr(o, '__dict__'): [w(v) for v in o.__dict__.values()]
        w(self); return p
    def f(self, x):
        x = self.emb.f(x[:, None, :] if x.ndim == 2 else x)
        for b in self.blks: x = b.f(x)
        return self.hd.f(self.fn.f(x[:, -1, :]))
    def b(self, dy):
        dy = self.fn.b(self.hd.b(dy))
        db = np.zeros((dy.shape[0], self.emb.x.shape[1], dy.shape[1]), "f4")
        db[:, -1, :] = dy
        for b in reversed(self.blks): db = b.b(db)
        self.emb.b(db)

class AW:
    def __init__(self, ps, lr=1e-3, wd=0.01):
        self.ps, self.lr, self.wd = ps, lr, wd
        self.m = [np.zeros_like(p.d) for p in ps]
        self.v = [np.zeros_like(p.d) for p in ps]
        self.t = 0
    def step(self):
        self.t += 1
        lt = self.lr * (1 - 0.999**self.t)**.5 / (1 - 0.9**self.t)
        for i, p in enumerate(self.ps):
            g = np.clip(p.g, -1, 1)
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * (g**2)
            p.d -= lt * (self.m[i] / (np.sqrt(self.v[i]) + 1e-8) + self.wd * p.d)
            p.g.fill(0)

def train():
    N, D, C, BS, E = 1024, 784, 10, 64, 50
    X, Y = np.random.randn(N, D).astype("f4"), np.random.randint(0, C, N)
    m = OMEGA(D, 128, C)
    opt = AW(m.ps, 3e-3)
    for e in range(E):
        idx = np.random.permutation(N)
        ls, ac = [], []
        for i in range(0, N, BS):
            xb, yb = X[idx[i:i+BS]], Y[idx[i:i+BS]]
            lg = m.f(xb)
            pr = np.exp(lg - lg.max(-1, keepdims=True))
            pr /= pr.sum(-1, keepdims=True)
            ls.append(-np.mean(np.log(pr[np.arange(len(yb)), yb] + 1e-12)))
            ac.append(np.mean(np.argmax(pr, -1) == yb))
            dl = pr.copy(); dl[np.arange(len(yb)), yb] -= 1
            m.b(dl / len(yb))
            opt.step()
        if (e+1) % 5 == 0: print(f"E {e+1} | L {np.mean(ls):.4f} | A {np.mean(ac):.4f}")

if __name__ == "__main__": train()