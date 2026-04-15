import numpy as np

def sg(x): return 1 / (1 + np.exp(-np.clip(x, -12, 12)))

class L:
    def __init__(self, i, o, s=None):
        self.W = np.random.randn(i, o).astype("f4") * (s or np.sqrt(2/i))
        self.b = np.zeros(o, "f4")
    def f(self, x):
        self.x = x
        return x @ self.W + self.b
    def bwd(self, dy):
        self.dW = self.x.reshape(-1, self.x.shape[-1]).T @ dy.reshape(-1, dy.shape[-1])
        self.db = dy.reshape(-1, dy.shape[-1]).sum(0)
        return (dy @ self.W.T).reshape(self.x.shape[:-1] + (self.W.shape[0],))

class RN:
    def __init__(self, d, e=1e-6):
        self.g, self.e = np.ones(d, "f4"), e
    def f(self, x):
        self.x = x
        self.v = np.mean(x**2, -1, keepdims=True)
        self.r = 1 / np.sqrt(self.v + self.e)
        return self.g * (x * self.r)
    def bwd(self, dy):
        xn = self.x * self.r
        self.dg = (dy * xn).reshape(-1, xn.shape[-1]).sum(0)
        dn = dy * self.g
        return self.r * (dn - xn * np.mean(dn * xn, -1, keepdims=True))

class RP:
    def __init__(self, d, m=4096):
        f = 1. / (10000 ** (np.arange(0, d, 2) / d))
        t = np.outer(np.arange(m), f)
        self.c, self.s = np.cos(t), np.sin(t)
    def apply(self, x, conj=False):
        s, d = x.shape[1], x.shape[-1]
        x1, x2 = x[..., :d//2], x[..., d//2:]
        c, sn = self.c[:s, None, :], self.s[:s, None, :]
        if conj: return np.concatenate([x1 * c + x2 * sn, x2 * c - x1 * sn], -1)
        return np.concatenate([x1 * c - x2 * sn, x2 * c + x1 * sn], -1)

class LC:
    def __init__(self, d):
        self.lp, self.gp, self.o = L(d, d*2), L(d, d*2), L(d*2, d)
    def f(self, x):
        self.ge, self.gr = self.lp.f(x), self.gp.f(x)
        self.sig = sg(self.ge)
        self.act = (self.ge * self.sig) * self.gr
        return self.o.f(self.act)
    def bwd(self, dy):
        da = self.o.bwd(dy)
        dge = da * self.gr * (self.sig * (1 + self.ge * (1 - self.sig)))
        dgr = da * (self.ge * self.sig)
        return self.lp.bwd(dge) + self.gp.bwd(dgr)

class GQA:
    def __init__(self, d, h=8, g=2):
        self.d, self.h, self.g, self.hd = d, h, g, d // h
        self.wq, self.wk, self.wv, self.wo = L(d, d), L(d, (h//g)*self.hd), L(d, (h//g)*self.hd), L(d, d)
        self.rp, self.sc = RP(self.hd), (d//h)**-0.5
    def f(self, x):
        b, s, _ = x.shape
        q, k, v = self.wq.f(x).reshape(b,s,self.h,self.hd), self.wk.f(x).reshape(b,s,self.h//self.g,self.hd), self.wv.f(x).reshape(b,s,self.h//self.g,self.hd)
        self.qr, self.kr, self.v_raw = self.rp.apply(q), self.rp.apply(k), v
        ke, ve = np.repeat(self.kr, self.g, 2), np.repeat(v, self.g, 2)
        at = np.einsum("bshd,bthd->bsht", self.qr, ke) * self.sc
        at -= at.max(-1, keepdims=True)
        self.p = np.exp(at); self.p /= self.p.sum(-1, keepdims=True) + 1e-12
        return self.wo.f(np.einsum("bsht,bthd->bshd", self.p, ve).reshape(b, s, -1))
    def bwd(self, dy):
        b, s = dy.shape[:2]
        dy_o = self.wo.bwd(dy).reshape(b, s, self.h, self.hd)
        ke, ve = np.repeat(self.kr, self.g, 2), np.repeat(self.v_raw, self.g, 2)
        dp = np.einsum("bshd,bthd->bsht", dy_o, ve)
        da = self.p * (dp - (self.p * dp).sum(-1, keepdims=True)) * self.sc
        dq = self.rp.apply(np.einsum("bsht,bthd->bshd", da, ke), True)
        dk = self.rp.apply(np.einsum("bsht,bshd->bthd", da, self.qr).reshape(b,s,self.h//self.g,self.g,self.hd).sum(3), True)
        dv = np.einsum("bsht,bshd->bthd", self.p, dy_o).reshape(b,s,self.h//self.g,self.g,self.hd).sum(3)
        return self.wq.bwd(dq.reshape(b,s,-1)) + self.wk.bwd(dk.reshape(b,s,-1)) + self.wv.bwd(dv.reshape(b,s,-1))

class MoE:
    def __init__(self, d, n=4, e=2):
        self.n, self.gate, self.ex = n, L(d, n), [[L(d, d*e), L(d*e, d), L(d, d*e)] for _ in range(n)]
    def f(self, x):
        self.x = x
        g = self.gate.f(x)
        self.pr = np.exp(g - g.max(-1, keepdims=True)); self.pr /= self.pr.sum(-1, keepdims=True)
        self.c, o = [], np.zeros_like(x)
        for i in range(self.n):
            w1, w2, w3 = self.ex[i]
            x1, x3 = w1.f(x), w3.f(x)
            s = sg(x1)
            act = (x1 * s) * x3
            self.c.append((x1, x3, s, act))
            o += self.pr[..., i:i+1] * w2.f(act)
        return o
    def bwd(self, dy):
        dx, dpr = np.zeros_like(self.x), np.zeros_like(self.pr)
        for i in range(self.n):
            x1, x3, s, act = self.c[i]
            dpr[..., i] = (dy * self.ex[i][1].f(act)).sum(-1)
            da = self.ex[i][1].bwd(dy * self.pr[..., i:i+1])
            dx += self.ex[i][0].bwd(da * x3 * (s * (1 + x1 * (1 - s)))) + self.ex[i][2].bwd(da * (x1 * s))
        return dx + self.gate.bwd(self.pr * (dpr - (self.pr * dpr).sum(-1, keepdims=True)))

class SB:
    def __init__(self, d):
        self.n1, self.at, self.n2, self.lc, self.n3, self.me = RN(d), GQA(d), RN(d), LC(d), RN(d), MoE(d)
    def f(self, x):
        x = x + self.at.f(self.n1.f(x))
        x = x + self.lc.f(self.n2.f(x))
        return x + self.me.f(self.n3.f(x))
    def bwd(self, dy):
        dy = dy + self.n3.bwd(self.me.bwd(dy))
        dy = dy + self.n2.bwd(self.lc.bwd(dy))
        return dy + self.n1.bwd(self.at.bwd(dy))

class ASI:
    def __init__(self, i, h, o, d=2):
        self.e, self.blks, self.ln, self.hd = L(i, h), [SB(h) for _ in range(d)], RN(h), L(h, o)
    def f(self, x):
        x = self.e.f(x[:, None] if x.ndim==2 else x)
        for b in self.blks: x = b.f(x)
        return self.hd.f(self.ln.f(x[:, -1]))
    def bwd(self, dy):
        dy = self.ln.bwd(self.hd.bwd(dy))
        dys = np.zeros((dy.shape[0], 1, dy.shape[1]), "f4")
        dys[:, -1] = dy
        for b in reversed(self.blks): dys = b.bwd(dys)
        self.e.bwd(dys)
    def params(self):
        res = []
        def g(o):
            if isinstance(o, (L, RN)): res.append(o)
            elif isinstance(o, list): [g(i) for i in o]
            elif hasattr(o, "__dict__"): [g(v) for k, v in o.__dict__.items() if k[0] != "_"]
        g(self); return list(set(res))

class Op:
    def __init__(self, p, lr=1e-3, b1=0.9, b2=0.999, wd=0.01):
        self.p, self.lr, self.b1, self.b2, self.wd, self.t = p, lr, b1, b2, wd, 0
        self.m = {id(x): [np.zeros_like(getattr(x, a)) for a in (["W","b"] if hasattr(x,"W") else ["g"])] for x in p}
        self.v = {id(x): [np.zeros_like(getattr(x, a)) for a in (["W","b"] if hasattr(x,"W") else ["g"])] for x in p}
    def step(self):
        self.t += 1
        lt = self.lr * min(1., self.t/100) * (0.5 * (1 + np.cos(min(self.t, 1000) * np.pi / 1000)))
        for x in self.p:
            at = ["W", "b"] if hasattr(x, "W") else ["g"]
            for i, a in enumerate(at):
                gr = getattr(x, "d"+a if a!="g" else "dg")
                m, v = self.m[id(x)], self.v[id(x)]
                m[i] = self.b1 * m[i] + (1 - self.b1) * gr
                v[i] = self.b2 * v[i] + (1 - self.b2) * (gr**2)
                mh, vh = m[i] / (1 - self.b1**self.t), v[i] / (1 - self.b2**self.t)
                pv = getattr(x, a); pv -= lt * (mh / (np.sqrt(vh) + 1e-8) + self.wd * pv)
                setattr(x, a, pv)

def train():
    N, D, C, BS, E = 1024, 784, 10, 32, 100
    X, Y = np.random.randn(N, D).astype("f4"), np.random.randint(0, C, N)
    m = ASI(D, 128, C); ps = m.params(); opt = Op(ps, 2e-3)
    for e in range(E):
        idx = np.random.permutation(N); ls, ac = 0, 0
        for i in range(0, N, BS):
            xb, yb = X[idx[i:i+BS]], Y[idx[i:i+BS]]
            l = m.f(xb); pr = np.exp(l - l.max(-1, 1, 1)); pr /= pr.sum(-1, 1, 1)
            ls += -np.log(pr[range(len(yb)), yb] + 1e-12).sum()
            ac += (pr.argmax(1) == yb).sum()
            dy = pr.copy(); dy[range(len(yb)), yb] -= 1; m.bwd(dy / len(yb))
            gn = np.sqrt(sum((getattr(p, "d"+a if a!="g" else "dg")**2).sum() for p in ps for a in (["W","b"] if hasattr(p,"W") else ["g"])))
            if gn > 1:
                for p in ps:
                    for a in (["W","b"] if hasattr(p,"W") else ["g"]):
                        k = "d"+a if a!="g" else "dg"; setattr(p, k, getattr(p, k)/gn)
            opt.step()
        if (e+1)%5==0: print(f"E {e+1:03d} | L: {ls/N:.4f} | A: {ac/N:.4f}")

if __name__ == "__main__": train()