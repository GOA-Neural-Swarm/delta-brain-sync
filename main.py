import numpy as np

class Ops:
    @staticmethod
    def sm(x):
        e = np.exp(x - np.max(x, -1, keepdims=1))
        return e / (e.sum(-1, keepdims=1) + 1e-9)

    @staticmethod
    def swi_f(x1, x2):
        return (x1 / (1 + np.exp(-np.clip(x1, -12, 12)))) * x2

    @staticmethod
    def swi_b(x1, x2, d):
        s = 1 / (1 + np.exp(-np.clip(x1, -12, 12)))
        ds1 = d * x2 * (s * (1 + x1 * (1 - s)))
        return ds1, d * (x1 * s)

class Lin:
    def __init__(self, i, o):
        self.W = (np.random.randn(i, o) * (2/i)**.5).astype("f")
        self.b = np.zeros(o, "f")
    def f(self, x):
        self.x = x
        return x @ self.W + self.b
    def bwd(self, d):
        self.dW = self.x.reshape(-1, self.x.shape[-1]).T @ d.reshape(-1, d.shape[-1])
        self.db = d.sum(tuple(range(d.ndim - 1)))
        return d @ self.W.T

class Norm:
    def __init__(self, d):
        self.g, self.e = np.ones(d, "f"), 1e-6
    def f(self, x):
        self.x = x
        self.v = 1 / np.sqrt(np.mean(x**2, -1, keepdims=1) + self.e)
        return self.g * (x * self.v)
    def bwd(self, d):
        nx = self.x * self.v
        self.dg = (d * nx).sum(tuple(range(d.ndim - 1)))
        dn = d * self.g
        return self.v * (dn - nx * np.mean(dn * nx, -1, keepdims=1))

class RoPE:
    def __init__(self, d, m=2048):
        f = 1 / (10000**(np.arange(0, d, 2)/d))
        fr = np.outer(np.arange(m), f)
        self.c, self.s = np.cos(fr), np.sin(fr)
    def a(self, x, r=0):
        s = x.shape[1]
        c, sn = self.c[:s][None, :, None, :], self.s[:s][None, :, None, :]
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        if r: return np.concatenate([x1*c + x2*sn, x2*c - x1*sn], -1)
        return np.concatenate([x1*c - x2*sn, x2*c + x1*sn], -1)

class Attn:
    def __init__(self, d, h=8, k=2):
        self.d, self.h, self.k, self.hd = d, h, k, d//h
        self.g = h//k
        self.wq, self.wk, self.wv, self.wo = Lin(d, d), Lin(d, k*self.hd), Lin(d, k*self.hd), Lin(d, d)
        self.rope, self.sc = RoPE(self.hd), (d//h)**-0.5
    def f(self, x):
        b, s, _ = x.shape
        q = self.wq.f(x).reshape(b, s, self.h, self.hd)
        k = self.wk.f(x).reshape(b, s, self.k, self.hd)
        v = self.wv.f(x).reshape(b, s, self.k, self.hd)
        self.q, self.kc, self.vc = self.rope.a(q), self.rope.a(k), v
        self.p = Ops.sm(np.einsum("bshd,bthd->bsht", self.q, np.repeat(self.kc, self.g, 2)) * self.sc)
        return self.wo.f(np.einsum("bsht,bthd->bshd", self.p, np.repeat(self.vc, self.g, 2)).reshape(b, s, self.d))
    def bwd(self, d):
        b, s, _ = d.shape
        do = self.wo.bwd(d).reshape(b, s, self.h, self.hd)
        kr, vr = np.repeat(self.kc, self.g, 2), np.repeat(self.vc, self.g, 2)
        dp = np.einsum("bshd,bthd->bsht", do, vr)
        da = self.p * (dp - (self.p * dp).sum(-1, keepdims=1)) * self.sc
        dq = self.rope.a(np.einsum("bsht,bthd->bshd", da, kr), 1)
        dk = self.rope.a(np.einsum("bsht,bshd->bthd", da, self.q).reshape(b, s, self.k, self.g, self.hd).sum(3), 1)
        dv = np.einsum("bsht,bshd->bthd", self.p, do).reshape(b, s, self.k, self.g, self.hd).sum(3)
        return self.wq.bwd(dq.reshape(b, s, -1)) + self.wk.bwd(dk.reshape(b, s, -1)) + self.wv.bwd(dv.reshape(b, s, -1))

class MLP:
    def __init__(self, d, e=4):
        self.w1, self.w2, self.w3 = Lin(d, d*e), Lin(d, d*e), Lin(d*e, d)
    def f(self, x):
        self.x1, self.x2 = self.w1.f(x), self.w2.f(x)
        return self.w3.f(Ops.swi_f(self.x1, self.x2))
    def bwd(self, d):
        ds1, ds2 = Ops.swi_b(self.x1, self.x2, self.w3.bwd(d))
        return self.w1.bwd(ds1) + self.w2.bwd(ds2)

class Block:
    def __init__(self, d):
        self.n, self.at, self.ml, self.gt = Norm(d), Attn(d), MLP(d), Lin(d, 2)
    def f(self, x):
        nx = self.n.f(x)
        self.pr = Ops.sm(self.gt.f(nx))
        self.og, self.om = self.at.f(nx), self.ml.f(nx)
        return x + self.pr[..., 0:1]*self.og + self.pr[..., 1:2]*self.om
    def bwd(self, d):
        nx, (p0, p1) = self.n.x, (self.pr[..., 0:1], self.pr[..., 1:2])
        da, dm = self.at.bwd(d * p0), self.ml.bwd(d * p1)
        dl = np.concatenate([(d*self.og).sum(-1, 1), (d*self.om).sum(-1, 1)], -1)
        dg = self.gt.bwd(self.pr * (dl - (self.pr * dl).sum(-1, 1)))
        return d + self.n.bwd(da + dm + dg)

class Model:
    def __init__(self, i, h, o, d=3):
        self.st, self.bl = Lin(i, h), [Block(h) for _ in range(d)]
        self.n, self.hd = Norm(h), Lin(h, o)
    def f(self, x):
        x = self.st.f(x[:, None] if x.ndim==2 else x)
        for b in self.bl: x = b.f(x)
        return self.hd.f(self.n.f(x[:, -1]))
    def bwd(self, d):
        d = self.n.bwd(self.hd.bwd(d))
        dz = np.zeros((d.shape[0], 1, d.shape[1]), "f")
        dz[:, -1] = d
        for b in reversed(self.bl): dz = b.bwd(dz)
        self.st.bwd(dz)
    def get_p(self):
        p = []
        def g(o):
            if isinstance(o, (Lin, Norm)): p.append(o)
            elif isinstance(o, list): [g(i) for i in o]
            elif hasattr(o, "__dict__"): [g(v) for k, v in o.__dict__.items() if k not in ("x", "v", "p", "pr", "og", "om", "q", "kc", "vc")]
        g(self); return list(set(p))

class Lion:
    def __init__(self, ps, lr=1e-4, b1=0.9, b2=0.99, wd=0.01):
        self.ps, self.lr, self.b1, self.b2, self.wd = ps, lr, b1, b2, wd
        self.m = {id(p): [np.zeros_like(getattr(p, a)) for a in (["W", "b"] if hasattr(p, "W") else ["g"])] for p in ps}
    def step(self, s=1.0):
        lr = self.lr * s
        for p in self.ps:
            at = ["W", "b"] if hasattr(p, "W") else ["g"]
            for i, a in enumerate(at):
                g, w = getattr(p, "d"+a if a!="g" else "dg"), getattr(p, a)
                m = self.m[id(p)][i]
                u = np.sign(self.b1 * m + (1 - self.b1) * g)
                w -= lr * (u + self.wd * w if a in ("W", "g") else u)
                self.m[id(p)][i] = self.b2 * m + (1 - self.b2) * g
                setattr(p, a, w)

if __name__ == "__main__":
    N, D, C, bs, eps = 2048, 784, 10, 64, 30
    X, Y = (np.random.randn(N, D)*.02).astype("f"), np.random.randint(0, C, N)
    m = Model(D, 128, C)
    ps = m.get_p()
    opt = Lion(ps, 3e-4, wd=0.1)
    for e in range(eps):
        idx = np.random.permutation(N)
        ls, acc = 0, 0
        lrs = (e+1)/5 if e<5 else 0.5*(1+np.cos(np.pi*(e-5)/(eps-5)))
        for i in range(0, N, bs):
            xb, yb = X[idx[i:i+bs]], Y[idx[i:i+bs]]
            p = Ops.sm(m.f(xb))
            ls -= np.log(p[range(len(yb)), yb] + 1e-10).sum()
            acc += (p.argmax(1) == yb).sum()
            do = p.copy(); do[range(len(yb)), yb] -= 1
            m.bwd(do / len(yb))
            gn = np.sqrt(sum((getattr(p, "dW", 0)**2).sum() + (getattr(p, "db", 0)**2).sum() + (getattr(p, "dg", 0)**2).sum() for p in ps))
            if gn > 1:
                for p in ps:
                    if hasattr(p, "W"): p.dW/=gn; p.db/=gn
                    else: p.dg/=gn
            opt.step(lrs)
        print(f"E {e+1:02} | L: {ls/N:.4f} | A: {acc/N:.4f} | LR: {opt.lr*lrs:.6f}")