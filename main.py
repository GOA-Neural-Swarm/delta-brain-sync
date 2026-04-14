import numpy as np

class Act:
    @staticmethod
    def swi(x):
        h = x.shape[-1] // 2
        a, b = x[..., :h], x[..., h:]
        s = 1 / (1 + np.exp(-np.clip(a, -12, 12)))
        return (a * s) * b
    @staticmethod
    def dswi(x, d):
        h = x.shape[-1] // 2
        a, b = x[..., :h], x[..., h:]
        s = 1 / (1 + np.exp(-np.clip(a, -12, 12)))
        sw = a * s
        return np.concatenate([d * b * (s + sw * (1 - s)), d * sw], -1)
    @staticmethod
    def ge(x): return 0.5 * x * (1 + np.tanh(0.79788 * (x + 0.0447 * x**3)))
    @staticmethod
    def dge(x, d):
        t = np.tanh(0.79788 * (x + 0.0447 * x**3))
        return d * (0.5 * (1 + t) + x * (0.39894 * np.exp(-0.5 * x**2)))

class Lin:
    def __init__(self, i, o):
        self.W, self.b = np.random.randn(i, o).astype("f") * (2/i)**.5, np.zeros(o, "f")
    def forward(self, x):
        self.x = x
        return x @ self.W + self.b
    def backward(self, d):
        self.dW = self.x.reshape(-1, self.x.shape[-1]).T @ d.reshape(-1, d.shape[-1])
        self.db = d.sum(axis=tuple(range(d.ndim - 1)))
        return d @ self.W.T

class Norm:
    def __init__(self, d, e=1e-6): self.g, self.e = np.ones(d, "f"), e
    def forward(self, x):
        self.x, self.v = x, 1 / (np.mean(x**2, -1, keepdims=True) + self.e)**.5
        return self.g * (x * self.v)
    def backward(self, d):
        nx = self.x * self.v
        self.dg = (d * nx).sum(axis=tuple(range(d.ndim - 1)))
        dn = (d * self.g)
        return self.v * (dn - nx * np.mean(dn * nx, -1, keepdims=True))

class Pos:
    def __init__(self, d, m=4096):
        f = 1 / (10000 ** (np.arange(0, d, 2) / d))
        fr = np.outer(np.arange(m), f)
        self.c, self.s = np.cos(fr), np.sin(fr)
    def apply(self, x, r=False):
        s, d2 = x.shape[1], x.shape[-1] // 2
        x1, x2, c, n = x[..., :d2], x[..., d2:], self.c[:s][None, :, None, :], self.s[:s][None, :, None, :]
        return np.concatenate([x1*c + x2*n, x2*c - x1*n], -1) if r else np.concatenate([x1*c - x2*n, x2*c + x1*n], -1)

class Attn:
    def __init__(self, d, h=8, k=2):
        self.d, self.h, self.k, self.hd = d, h, k, d // h
        self.g = h // k
        self.wq, self.wk, self.wv, self.wo = Lin(d, d), Lin(d, k*self.hd), Lin(d, k*self.hd), Lin(d, d)
        self.rope, self.sc = Pos(self.hd), (d // h)**-0.5
    def forward(self, x):
        b, s, _ = x.shape
        q, k, v = self.wq.forward(x).reshape(b, s, self.h, self.hd), self.wk.forward(x).reshape(b, s, self.k, self.hd), self.wv.forward(x).reshape(b, s, self.k, self.hd)
        self.q, self.k_c, self.v_c = self.rope.apply(q), self.rope.apply(k), v
        kr, vr = np.repeat(self.k_c, self.g, 2), np.repeat(v, self.g, 2)
        at = np.einsum("bshd,bthd->bsht", self.q, kr) * self.sc
        self.p = (e := np.exp(at - at.max(-1, keepdims=True))) / (e.sum(-1, keepdims=True) + 1e-10)
        return self.wo.forward(np.einsum("bsht,bthd->bshd", self.p, vr).reshape(b, s, self.d))
    def backward(self, d):
        b, s, _ = d.shape
        do = self.wo.backward(d).reshape(b, s, self.h, self.hd)
        kr, vr = np.repeat(self.k_c, self.g, 2), np.repeat(self.v_c, self.g, 2)
        dv, dp = np.einsum("bsht,bshd->bthd", self.p, do), np.einsum("bshd,bthd->bsht", do, vr)
        da = self.p * (dp - (self.p * dp).sum(-1, keepdims=True)) * self.sc
        dq = self.rope.apply(np.einsum("bsht,bthd->bshd", da, kr), True)
        dk = self.rope.apply(np.einsum("bsht,bshd->bthd", da, self.q).reshape(b, s, self.k, self.g, self.hd).sum(3), True)
        dv = dv.reshape(b, s, self.k, self.g, self.hd).sum(3)
        return self.wq.backward(dq.reshape(b, s, -1)) + self.wk.backward(dk.reshape(b, s, -1)) + self.wv.backward(dv.reshape(b, s, -1))

class Logic:
    def __init__(self, d):
        self.ge, self.gr, self.gt = [Lin(d, d*2), Lin(d, d)], [Lin(d, d), Lin(d, d)], Lin(d, 2)
    def forward(self, x):
        self.o_ge, self.o_gr = self.ge[1].forward(Act.swi(self.ge[0].forward(x))), self.gr[1].forward(Act.ge(self.gr[0].forward(x)))
        g = self.gt.forward(x)
        self.p = (e := np.exp(g - g.max(-1, keepdims=True))) / (e.sum(-1, keepdims=True) + 1e-10)
        return self.p[..., :1] * self.o_ge + self.p[..., 1:2] * self.o_gr
    def backward(self, d):
        dp = np.stack([(d * self.o_ge).sum(-1), (d * self.o_gr).sum(-1)], -1)
        dg = self.p * (dp - (self.p * dp).sum(-1, keepdims=True))
        dx = self.gt.backward(dg)
        dx += self.ge[0].backward(Act.dswi(self.ge[0].x, self.ge[1].backward(d * self.p[..., :1])))
        dx += self.gr[0].backward(Act.dge(self.gr[0].x, self.gr[1].backward(d * self.p[..., 1:2])))
        return dx

class MoE:
    def __init__(self, d, n=4, k=2):
        self.d, self.n, self.k, self.gt = d, n, k, Lin(d, n)
        self.ex = [[Lin(d, d*2), Lin(d, d)] for _ in range(n)]
    def forward(self, x):
        s, xf = x.shape, x.reshape(-1, self.d)
        p_all = (e := np.exp((l := self.gt.forward(xf)) - l.max(-1, keepdims=True))) / (e.sum(-1, keepdims=True) + 1e-10)
        self.idx = np.argsort(p_all, -1)[:, -self.k:]
        self.p = np.take_along_axis(p_all, self.idx, -1)
        self.p /= self.p.sum(-1, keepdims=True) + 1e-10
        out, self.ch = np.zeros_like(xf), []
        for i in range(self.n):
            m = np.any(self.idx == i, -1)
            if not np.any(m): self.ch.append(None); continue
            pi = self.p[m, np.where(self.idx[m] == i)[1], None]
            h = Act.swi(self.ex[i][0].forward(xf[m]))
            y = self.ex[i][1].forward(h)
            out[m] += y * pi
            self.ch.append((m, pi, h, y))
        return out.reshape(s)
    def backward(self, d):
        df, dx, dpf = d.reshape(-1, self.d), np.zeros_like(d.reshape(-1, self.d)), np.zeros((d.reshape(-1, self.d).shape[0], self.n))
        for i in range(self.n):
            if self.ch[i]:
                m, pi, h, y = self.ch[i]
                dpf[m, i], dy = (df[m] * y).sum(-1), df[m] * pi
                dx[m] += self.ex[i][0].backward(Act.dswi(self.ex[i][0].x, self.ex[i][1].backward(dy)))
        lp = np.zeros((df.shape[0], self.n))
        np.put_along_axis(lp, self.idx, self.p, -1)
        return (dx + self.gt.backward(lp * (dpf - (lp * dpf).sum(-1, keepdims=True)))).reshape(d.shape)

class Blk:
    def __init__(self, d): self.n, self.a, self.l, self.m = [Norm(d) for _ in range(3)], Attn(d), Logic(d), MoE(d)
    def forward(self, x):
        x = x + self.a.forward(self.n[0].forward(x))
        x = x + self.l.forward(self.n[1].forward(x))
        return x + self.m.forward(self.n[2].forward(x))
    def backward(self, d):
        d = d + self.n[2].backward(self.m.backward(d))
        d = d + self.n[1].backward(self.l.backward(d))
        return d + self.n[0].backward(self.a.backward(d))

class OMEGA:
    def __init__(self, i=784, h=128, o=10, d=2):
        self.st, self.bl, self.nm, self.hd = Lin(i, h), [Blk(h) for _ in range(d)], Norm(h), Lin(h, o)
    def forward(self, x):
        x = self.st.forward(x)[:, None, :]
        for b in self.bl: x = b.forward(x)
        return self.hd.forward(self.nm.forward(x[:, 0, :]))
    def backward(self, d):
        d = self.nm.backward(self.hd.backward(d))[:, None, :]
        for b in reversed(self.bl): d = b.backward(d)
        self.st.backward(d[:, 0, :])
    def par(self):
        p = []
        def f(o):
            if isinstance(o, (Lin, Norm)): p.append(o)
            elif isinstance(o, list): [f(i) for i in o]
            elif hasattr(o, "__dict__"): [f(v) for v in o.__dict__.values()]
        f(self); return p

class Lion:
    def __init__(self, p, lr=1e-4, b1=0.9, b2=0.99, wd=0.01):
        self.p, self.lr, self.b1, self.b2, self.wd = p, lr, b1, b2, wd
        self.m = [np.zeros_like(getattr(x, "W", getattr(x, "g", 0))) for x in p]
        self.mb = [np.zeros_like(x.b) if hasattr(x, "b") else None for x in p]
    def step(self):
        for i, x in enumerate(self.p):
            if hasattr(x, "W"):
                for a, m in [("W", self.m), ("b", self.mb)]:
                    if m[i] is None: continue
                    g, w = getattr(x, "d"+a), getattr(x, a)
                    u = np.sign(self.b1 * m[i] + (1-self.b1) * g)
                    w -= self.lr * (u + self.wd * w if a == "W" else u)
                    m[i] = self.b2 * m[i] + (1-self.b2) * g
                    setattr(x, a, w)
            else:
                u = np.sign(self.b1 * self.m[i] + (1-self.b1) * x.dg)
                x.g -= self.lr * (u + self.wd * x.g)
                self.m[i] = self.b2 * self.m[i] + (1-self.b2) * x.dg

def train():
    N, D, C = 2048, 784, 10
    X, Y = np.random.randn(N, D).astype("f"), np.random.randint(0, C, N)
    m = OMEGA(D, 128, C, 2)
    p = m.par()
    opt = Lion(p, 2e-4)
    for e in range(100):
        idx = np.random.permutation(N)
        l_s, a_s = 0, 0
        for i in range(0, N, 64):
            xb, yb = X[idx[i:i+64]], Y[idx[i:i+64]]
            pr = (ex := np.exp((lg := m.forward(xb)) - lg.max(1, 1))) / (ex.sum(1, 1) + 1e-10)
            l_s += -np.log(pr[range(len(yb)), yb] + 1e-10).mean() * len(yb)
            a_s += (pr.argmax(1) == yb).sum()
            do = pr.copy(); do[range(len(yb)), yb] -= 1
            m.backward(do / len(yb))
            gn = np.sqrt(sum((getattr(x, "dW", 0)**2).sum() + (getattr(x, "db", 0)**2).sum() + (getattr(x, "dg", 0)**2).sum() for x in p))
            if gn > 1:
                for x in p:
                    if hasattr(x, "dW"): x.dW /= gn; x.db /= gn
                    if hasattr(x, "dg"): x.dg /= gn
            opt.step()
        if (e+1) % 10 == 0: print(f"E {e+1} | L: {l_s/N:.3f} | A: {a_s/N:.3f}")

if __name__ == "__main__": train()