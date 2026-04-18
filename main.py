import numpy as np

class T:
    def __init__(self, d, n=""):
        self.data = np.ascontiguousarray(d.astype("f4"))
        self.grad = np.zeros_like(self.data)

class M:
    def params(self):
        p = []
        for v in self.__dict__.values():
            if isinstance(v, T): p.append(v)
            elif isinstance(v, M): p.extend(v.params())
            elif isinstance(v, list):
                for i in v:
                    if isinstance(i, (M, T)): p.extend(i.params() if isinstance(i, M) else [i])
        return p

class L(M):
    def __init__(self, i, o, b=0):
        self.w = T(np.random.randn(i, o) * (2/i)**.5)
        self.b = T(np.zeros(o)) if b else None
    def f(self, x):
        self.x = x
        return x @ self.w.data + (self.b.data if self.b else 0)
    def b(self, dy):
        xf, df = self.x.reshape(-1, self.x.shape[-1]), dy.reshape(-1, dy.shape[-1])
        self.w.grad += xf.T @ df
        if self.b: self.b.grad += df.sum(0)
        return (dy @ self.w.data.T).reshape(self.x.shape)

class R(M):
    def __init__(self, d, e=1e-6): self.g, self.e = T(np.ones(d)), e
    def f(self, x):
        self.x, self.v = x, np.mean(x**2, -1, keepdims=1)
        self.i = (self.v + self.e)**-.5
        self.nx = x * self.i
        return self.g.data * self.nx
    def b(self, dy):
        dg = dy * self.g.data
        self.g.grad += np.sum(dy * self.nx, axis=tuple(range(dy.ndim - 1)))
        return (dg - self.nx * np.mean(dg * self.nx, -1, keepdims=1)) * self.i

class S(M):
    def f(self, x):
        self.x = x
        self.g, self.v = np.split(x, 2, -1)
        self.s = 1 / (1 + np.exp(-np.clip(self.g, -15, 15)))
        return (self.g * self.s) * self.v
    def b(self, dy):
        ds, dv = dy * self.v, dy * (self.g * self.s)
        dg = ds * self.s * (1 + self.g * (1 - self.s))
        return np.concatenate([dg, dv], -1)

class G(M):
    def __init__(self, d, h=8, g=2):
        self.d, self.h, self.g, self.hd = d, h, g, d // h
        self.wq, self.wk, self.wv, self.wo = L(d, d), L(d, (h//g)*self.hd), L(d, (h//g)*self.hd), L(d, d)
        self.sc = self.hd**-.5
    def _rp(self, t, inv=0):
        b, s, h, d = t.shape
        f = 10000**-(np.arange(0, d, 2)/d)
        a = np.arange(s)[:, None] * f
        c, n = np.cos(a), np.sin(a) * (-1 if inv else 1)
        r, i = t[..., ::2], t[..., 1::2]
        o = np.empty_like(t)
        o[..., ::2], o[..., 1::2] = r*c[:, None, :]-i*n[:, None, :], r*n[:, None, :]+i*c[:, None, :]
        return o
    def f(self, x):
        b, s, _ = x.shape
        q, k, v = self.wq.f(x).reshape(b, s, self.h, self.hd), self.wk.f(x).reshape(b, s, self.h//self.g, self.hd), self.wv.f(x).reshape(b, s, self.h//self.g, self.hd)
        self.qr, self.kr, self.vr = self._rp(q), self._rp(k), v
        kr, vr = np.repeat(self.kr, self.g, 2), np.repeat(self.vr, self.g, 2)
        at = np.einsum("bshd,bthd->bsht", self.qr, kr) * self.sc
        self.p = (e := np.exp(at - at.max(-1, keepdims=1))) / (e.sum(-1, keepdims=1) + 1e-12)
        return self.wo.f(np.einsum("bsht,bthd->bshd", self.p, vr).reshape(b, s, -1))
    def b(self, dy):
        b, s = dy.shape[:2]
        dy_wo = self.wo.b(dy).reshape(b, s, self.h, self.hd)
        kr, vr = np.repeat(self.kr, self.g, 2), np.repeat(self.vr, self.g, 2)
        dp = np.einsum("bshd,bthd->bsht", dy_wo, vr)
        da = self.p * (dp - (self.p * dp).sum(-1, keepdims=1)) * self.sc
        dqr, dkr = np.einsum("bsht,bthd->bshd", da, kr), np.einsum("bsht,bshd->bthd", da, self.qr).reshape(b, s, self.h//self.g, self.g, self.hd).sum(3)
        dvr = np.einsum("bsht,bshd->bthd", self.p, dy_wo).reshape(b, s, self.h//self.g, self.g, self.hd).sum(3)
        return self.wq.b(self._rp(dqr, 1).reshape(b, s, -1)) + self.wk.b(self._rp(dkr, 1).reshape(b, s, -1)) + self.wv.b(dvr.reshape(b, s, -1))

class C(M):
    def __init__(self, d):
        self.p1, self.p2, self.gt = [L(d, d*4), S(), L(d*2, d)], [L(d, d*4), S(), L(d*2, d)], L(d, 2)
    def f(self, x):
        self.x = x
        self.o1 = self.p1[2].f(self.p1[1].f(self.p1[0].f(x)))
        self.o2 = self.p2[2].f(self.p2[1].f(self.p2[0].f(x)))
        lg = self.gt.f(x)
        self.p = (e := np.exp(lg - lg.max(-1, keepdims=1))) / (e.sum(-1, keepdims=1) + 1e-12)
        return self.p[..., :1] * self.o1 + self.p[..., 1:] * self.o2
    def b(self, dy):
        dg, dq = dy * self.p[..., :1], dy * self.p[..., 1:]
        dp = np.stack([(dy * self.o1).sum(-1), (dy * self.o2).sum(-1)], -1)
        dx1 = self.p1[0].b(self.p1[1].b(self.p1[2].b(dg)))
        dx2 = self.p2[0].b(self.p2[1].b(self.p2[2].b(dq)))
        return dx1 + dx2 + self.gt.b(dp - dp.mean(-1, keepdims=1))

class E(M):
    def __init__(self, d, n=4, k=2):
        self.d, self.n, self.k, self.gt = d, n, k, L(d, n)
        self.ex = [[L(d, d*4), S(), L(d*2, d)] for _ in range(n)]
    def f(self, x):
        self.sh = x.shape
        xf = x.reshape(-1, self.d)
        lg = self.gt.f(xf)
        p = (e := np.exp(lg - lg.max(-1, keepdims=1))) / (e.sum(-1, keepdims=1) + 1e-12)
        self.idx = np.argsort(p, -1)[:, -self.k:]
        self.w = np.take_along_axis(p, self.idx, -1)
        self.w /= self.w.sum(-1, keepdims=1) + 1e-12
        out, self.ca = np.zeros_like(xf), []
        for i in range(self.n):
            m = np.any(self.idx == i, -1)
            if not np.any(m): self.ca.append(0); continue
            ps = np.where(self.idx[m] == i)[1]
            h1 = self.ex[i][0].f(xf[m]); h2 = self.ex[i][1].f(h1); h3 = self.ex[i][2].f(h2)
            out[m] += h3 * self.w[m, ps][:, None]
            self.ca.append((m, ps, h3))
        return out.reshape(self.sh)
    def b(self, dy):
        dyf = dy.reshape(-1, self.d)
        dx, dg = np.zeros((dyf.shape[0], self.d)), np.zeros((dyf.shape[0], self.n))
        for i in range(self.n):
            if not self.ca[i]: continue
            m, ps, h3 = self.ca[i]
            dg[m, i] = (dyf[m] * h3).sum(-1)
            dx[m] += self.ex[i][0].b(self.ex[i][1].b(self.ex[i][2].b(dyf[m] * self.w[m, ps][:, None])))
        return (dx + self.gt.b(dg - dg.mean(-1, keepdims=1))).reshape(self.sh)

class B(M):
    def __init__(self, d):
        self.n1, self.at, self.n2, self.cc, self.n3, self.ff = R(d), G(d), R(d), C(d), R(d), E(d)
    def f(self, x):
        x = x + self.at.f(self.n1.f(x))
        x = x + self.cc.f(self.n2.f(x))
        return x + self.ff.f(self.n3.f(x))
    def b(self, dy):
        dy = dy + self.ff.b(self.n3.b(dy))
        dy = dy + self.cc.b(self.n2.b(dy))
        return dy + self.at.b(self.n1.b(dy))

class OMEGA(M):
    def __init__(self, di, dm, do, d=4):
        self.em = L(di, dm)
        self.bl = [B(dm) for _ in range(d)]
        self.no, self.hd = R(dm), L(dm, do)
    def f(self, x):
        x = self.em.f(x[:, None] if x.ndim == 2 else x)
        for b in self.bl: x = b.f(x)
        return self.hd.f(self.no.f(x[:, -1]))
    def b(self, dy):
        dy = self.no.b(self.hd.b(dy))
        db = np.zeros((dy.shape[0], self.em.x.shape[1], dy.shape[1]))
        db[:, -1] = dy
        for b in reversed(self.bl): db = b.b(db)
        self.em.b(db)

class Adam:
    def __init__(self, p, lr=1e-3, wd=0.01):
        self.p, self.lr, self.wd = p, lr, wd
        self.m, self.v, self.t = [np.zeros_like(i.data) for i in p], [np.zeros_like(i.data) for i in p], 0
    def step(self):
        self.t += 1
        lt = self.lr * ((1-0.999**self.t)**.5 / (1-0.9**self.t))
        for i, pt in enumerate(self.p):
            g = np.clip(pt.grad, -1, 1)
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * (g**2)
            pt.data -= lt * (self.m[i] / (self.v[i]**.5 + 1e-8) + self.wd * pt.data)
            pt.grad.fill(0)

def train():
    N, D, C_sz, BS, E_ep = 2048, 784, 10, 128, 100
    X, Y = np.random.randn(N, D).astype("f4"), np.random.randint(0, C_sz, N)
    model = OMEGA(D, 256, C_sz, d=3)
    opt = Adam(model.params(), 1e-3, 0.05)
    for e in range(E_ep):
        idx, ls, ac = np.random.permutation(N), [], []
        for i in range(0, N, BS):
            xb, yb = X[idx[i:i+BS]], Y[idx[i:i+BS]]
            lg = model.f(xb)
            pr = (p := np.exp(lg - lg.max(-1, keepdims=1))) / (p.sum(-1, keepdims=1) + 1e-12)
            ls.append(-np.mean(np.log(pr[range(len(yb)), yb] + 1e-12)))
            ac.append(np.mean(pr.argmax(-1) == yb))
            dl = pr.copy(); dl[range(len(yb)), yb] -= 1
            model.b(dl / len(yb))
            opt.step()
        if (e + 1) % 10 == 0: print(f"E {e+1:03} | L: {np.mean(ls):.4f} | A: {np.mean(ac):.4f}")

if __name__ == "__main__": train()