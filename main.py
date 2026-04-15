import numpy as np

class L:
    def __init__(self, i, o):
        self.W = (np.random.randn(i, o) * (2/i)**.5).astype('f4')
        self.b = np.zeros(o, 'f4')
    def f(self, x):
        self.x = x
        return x @ self.W + self.b
    def bwd(self, dy):
        self.dW = self.x.reshape(-1, self.x.shape[-1]).T @ dy.reshape(-1, dy.shape[-1])
        self.db = dy.sum(axis=tuple(range(dy.ndim-1)))
        return dy @ self.W.T

class RN:
    def __init__(self, d, e=1e-6): self.g, self.e = np.ones(d, 'f4'), e
    def f(self, x):
        self.x = x
        self.r = np.sqrt(np.mean(x**2, -1, keepdims=True) + self.e)
        return self.g * (x / self.r)
    def bwd(self, dy):
        xn = self.x / self.r
        self.dg = (dy * xn).sum(axis=tuple(range(dy.ndim-1)))
        dn = (dy * self.g)
        return (dn - xn * np.mean(dn * xn, -1, keepdims=True)) / self.r

class R:
    def __init__(self, d, m=2048):
        f = 1./(10000**(np.arange(0, d, 2)/d))
        t = np.outer(np.arange(m), f)
        self.c, self.s = np.cos(t), np.sin(t)
    def a(self, x, conj=False):
        b, s, h, d = x.shape
        x1, x2 = x[..., :d//2], x[..., d//2:]
        c, sn = self.c[:s, None, :], self.s[:s, None, :]
        if conj: return np.concatenate([x1*c+x2*sn, x2*c-x1*sn], -1)
        return np.concatenate([x1*c-x2*sn, x2*c+x1*sn], -1)

class G:
    def __init__(self, d, h=8, g=2):
        self.d, self.h, self.g, self.hd = d, h, g, d//h
        self.wq, self.wk, self.wv, self.wo = L(d, d), L(d, d//g), L(d, d//g), L(d, d)
        self.rope, self.sc = R(self.hd), (d//h)**-0.5
    def f(self, x):
        b, s, _ = x.shape
        q = self.wq.f(x).reshape(b, s, self.h, self.hd)
        k = self.wk.f(x).reshape(b, s, self.h//self.g, self.hd)
        v = self.wv.f(x).reshape(b, s, self.h//self.g, self.hd)
        self.qr, self.kr, self.v_v = self.rope.a(q), self.rope.a(k), v
        ke, ve = np.repeat(self.kr, self.g, 2), np.repeat(v, self.g, 2)
        at = np.einsum("bshd,bthd->bsht", self.qr, ke) * self.sc
        e = np.exp(at - at.max(-1, keepdims=True))
        self.p = e / (e.sum(-1, keepdims=True) + 1e-12)
        return self.wo.f(np.einsum("bsht,bthd->bshd", self.p, ve).reshape(b, s, -1))
    def bwd(self, dy):
        b, s, _ = dy.shape
        dy_wo = self.wo.bwd(dy).reshape(b, s, self.h, self.hd)
        ke, ve = np.repeat(self.kr, self.g, 2), np.repeat(self.v_v, self.g, 2)
        dp = np.einsum("bshd,bthd->bsht", dy_wo, ve)
        da = self.p * (dp - (self.p * dp).sum(-1, keepdims=True)) * self.sc
        dq = self.rope.a(np.einsum("bsht,bthd->bshd", da, ke), 1)
        dk_e = np.einsum("bsht,bshd->bthd", da, self.qr)
        dv_e = np.einsum("bsht,bshd->bthd", self.p, dy_wo)
        dk = self.rope.a(dk_e.reshape(b, s, self.h//self.g, self.g, self.hd).sum(3), 1)
        dv = dv_e.reshape(b, s, self.h//self.g, self.g, self.hd).sum(3)
        return self.wq.bwd(dq.reshape(b, s, -1)) + self.wk.bwd(dk.reshape(b, s, -1)) + self.wv.bwd(dv.reshape(b, s, -1))

class S:
    def __init__(self, d, e=4): self.w1, self.w2, self.w3 = L(d, d*e), L(d*e, d), L(d, d*e)
    def f(self, x):
        self.x1, self.x3 = self.w1.f(x), self.w3.f(x)
        self.sig = 1 / (1 + np.exp(-np.clip(self.x1, -12, 12)))
        self.sw = self.x1 * self.sig
        return self.w2.f(self.sw * self.x3)
    def bwd(self, dy):
        dw2 = self.w2.bwd(dy)
        dx3, dsw = dw2 * self.sw, dw2 * self.x3
        dx1 = dsw * (self.sig + self.x1 * self.sig * (1 - self.sig))
        return self.w1.bwd(dx1) + self.w3.bwd(dx3)

class M:
    def __init__(self, d, n=4, ed=4):
        self.g = L(d, n)
        self.ex = [S(d, ed) for _ in range(n)]
    def f(self, x):
        self.b, self.s, self.d = x.shape
        lg = self.g.f(x)
        ex = np.exp(lg - lg.max(-1, 1, keepdims=1))
        self.pr = ex / ex.sum(-1, keepdims=1)
        out, self.eo = np.zeros_like(x), []
        for i, e in enumerate(self.ex):
            o = e.f(x)
            self.eo.append(o)
            out += self.pr[..., i:i+1] * o
        return out
    def bwd(self, dy):
        dx, dp = np.zeros((self.b, self.s, self.d), 'f4'), np.zeros_like(self.pr)
        for i, e in enumerate(self.ex):
            dp[..., i] = (dy * self.eo[i]).sum(-1)
            dx += e.bwd(dy * self.pr[..., i:i+1])
        dl = self.pr * (dp - (self.pr * dp).sum(-1, 1, keepdims=1))
        return dx + self.g.bwd(dl)

class B:
    def __init__(self, d): self.n1, self.at, self.n2, self.ff = RN(d), G(d), RN(d), M(d)
    def f(self, x):
        x = x + self.at.f(self.n1.f(x))
        return x + self.ff.f(self.n2.f(x))
    def bwd(self, dy):
        dy = dy + self.n2.bwd(self.ff.bwd(dy))
        return dy + self.n1.bwd(self.at.bwd(dy))

class OMEGA_ASI:
    def __init__(self, i, h, o, d=4):
        self.emb = L(i, h)
        self.blks = [B(h) for _ in range(d)]
        self.ln = RN(h)
        self.head = L(h, o)
    def f(self, x):
        x = self.emb.f(x[:, None] if x.ndim == 2 else x)
        for b in self.blks: x = b.f(x)
        return self.head.f(self.ln.f(x[:, -1]))
    def bwd(self, dy):
        dy = self.ln.bwd(self.head.bwd(dy))
        dys = np.zeros((dy.shape[0], 1, dy.shape[1]), 'f4')
        dys[:, -1] = dy
        for b in reversed(self.blks): dys = b.bwd(dys)
        self.emb.bwd(dys)
    def params(self):
        p = []
        def g(o):
            if isinstance(o, (L, RN)): p.append(o)
            elif isinstance(o, list): [g(i) for i in o]
            elif hasattr(o, "__dict__"): [g(v) for k, v in o.__dict__.items() if k[0]!='_']
        g(self); return list(set(p))

class AdamW:
    def __init__(self, p, lr=1e-3, b1=.9, b2=.999, wd=.01):
        self.p, self.lr, self.b1, self.b2, self.wd, self.t = p, lr, b1, b2, wd, 0
        self.m = {id(x): [np.zeros_like(getattr(x, a)) for a in (['W','b'] if hasattr(x,'W') else ['g'])] for x in p}
        self.v = {id(x): [np.zeros_like(getattr(x, a)) for a in (['W','b'] if hasattr(x,'W') else ['g'])] for x in p}
    def step(self):
        self.t += 1
        for x in self.p:
            at = ['W', 'b'] if hasattr(x, 'W') else ['g']
            for i, a in enumerate(at):
                gr = getattr(x, 'd'+a if a!='g' else 'dg')
                m, v = self.m[id(x)][i], self.v[id(x)][i]
                m[:] = self.b1 * m + (1-self.b1) * gr
                v[:] = self.b2 * v + (1-self.b2) * (gr**2)
                mh, vh = m/(1-self.b1**self.t), v/(1-self.b2**self.t)
                pv = getattr(x, a)
                pv -= self.lr * (mh/(np.sqrt(vh)+1e-8) + self.wd * pv)
                setattr(x, a, pv)

def train():
    N, D, C, BS, EP = 1024, 784, 10, 64, 20
    X, Y = (np.random.randn(N, D)*.01).astype('f4'), np.random.randint(0, C, N)
    m = OMEGA_ASI(D, 64, C, d=2)
    ps = m.params()
    opt = AdamW(ps)
    for e in range(EP):
        idx = np.random.permutation(N)
        ls, ac = 0, 0
        for i in range(0, N, BS):
            xb, yb = X[idx[i:i+BS]], Y[idx[i:i+BS]]
            lg = m.f(xb)
            pr = np.exp(lg - lg.max(-1, keepdims=1))
            pr /= pr.sum(-1, keepdims=1)
            ls += -np.log(pr[range(len(yb)), yb] + 1e-12).sum()
            ac += (pr.argmax(1) == yb).sum()
            dy = pr.copy(); dy[range(len(yb)), yb] -= 1
            m.bwd(dy/len(yb))
            gn = np.sqrt(sum((getattr(p,'d'+a if a!='g' else 'dg')**2).sum() for p in ps for a in (['W','b'] if hasattr(p,'W') else ['g'])))
            if gn > 1:
                for p in ps:
                    for a in (['W','b'] if hasattr(p,'W') else ['g']):
                        k = 'd'+a if a!='g' else 'dg'; setattr(p, k, getattr(p, k)/gn)
            opt.step()
        print(f"E{e+1} | L: {ls/N:.3f} | A: {ac/N:.3f}")

if __name__ == "__main__": train()