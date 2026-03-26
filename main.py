import numpy as np
import time

class P:
    def __init__(self, d, n=""):
        self.d = d.astype(np.float32)
        self.g = np.zeros_like(d)
        self.m = np.zeros_like(d)
        self.v = np.zeros_like(d)

class M:
    def __init__(self): self.tr = True
    def forward(self, x): pass
    def backward(self, g): pass
    def get_p(self):
        ps = []
        for v in self.__dict__.values():
            if isinstance(v, P): ps.append(v)
            elif isinstance(v, M): ps.extend(v.get_p())
            elif isinstance(v, list):
                for i in v: 
                    if isinstance(i, M): ps.extend(i.get_p())
        return ps

class L(M):
    def __init__(self, i, o):
        super().__init__()
        s = np.sqrt(2./i)
        self.w = P(np.random.randn(i, o)*s)
        self.b = P(np.zeros((1, o)))
    def forward(self, x):
        self.x = x
        return x @ self.w.d + self.b.d
    def backward(self, g):
        self.w.g = self.x.T @ g
        self.b.g = g.sum(0, keepdims=True)
        return g @ self.w.d.T

class LN(M):
    def __init__(self, d, e=1e-6):
        super().__init__()
        self.g, self.b, self.e = P(np.ones((1, d))), P(np.zeros((1, d))), e
    def forward(self, x):
        self.m = x.mean(-1, keepdims=True)
        self.v = x.var(-1, keepdims=True)
        self.s = np.sqrt(self.v + self.e)
        self.h = (x - self.m) / self.s
        return self.g.d * self.h + self.b.d
    def backward(self, g):
        n, d = g.shape
        self.g.g = (g * self.h).sum(0, keepdims=True)
        self.b.g = g.sum(0, keepdims=True)
        dxh = g * self.g.d
        return (d*dxh - dxh.sum(-1, keepdims=True) - self.h*(dxh*self.h).sum(-1, keepdims=True))/(d*self.s)

class GELU(M):
    def forward(self, x):
        self.x = x
        self.ti = 0.79788 * (x + 0.044715 * x**3)
        self.to = np.tanh(self.ti)
        return 0.5 * x * (1 + self.to)
    def backward(self, g):
        dt = (1 - self.to**2) * 0.79788 * (1 + 0.134145 * self.x**2)
        return g * (0.5 * (1 + self.to) + 0.5 * self.x * dt)

class Drop(M):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p
    def forward(self, x):
        if not self.tr or self.p == 0: return x
        self.mk = (np.random.rand(*x.shape) > self.p) / (1-self.p)
        return x * self.mk
    def backward(self, g): return g * self.mk if self.tr and self.p > 0 else g

class Res(M):
    def __init__(self, d):
        super().__init__()
        self.ln, self.l1, self.act, self.l2, self.dr = LN(d), L(d, d*4), GELU(), L(d*4, d), Drop()
    def forward(self, x):
        self.r = x
        o = self.ln.forward(x)
        o = self.l1.forward(o)
        o = self.act.forward(o)
        o = self.l2.forward(o)
        return self.dr.forward(o) + self.r
    def backward(self, g):
        dx = self.dr.backward(g)
        dx = self.l2.backward(dx)
        dx = self.act.backward(dx)
        dx = self.l1.backward(dx)
        return self.ln.backward(dx) + g

class Opt:
    def __init__(self, p, lr=1e-3, b=(0.9, 0.999), e=1e-8, wd=0.01):
        self.p, self.lr, self.b, self.e, self.wd, self.t = p, lr, b, e, wd, 0
    def step(self):
        self.t += 1
        lr_t = self.lr * (np.sqrt(1 - self.b[1]**self.t) / (1 - self.b[0]**self.t))
        for p in self.p:
            if self.wd > 0: p.d -= self.lr * self.wd * p.d
            p.m = self.b[0] * p.m + (1-self.b[0]) * p.g
            p.v = self.b[1] * p.v + (1-self.b[1]) * (p.g**2)
            p.d -= lr_t * p.m / (np.sqrt(p.v) + self.e)

class Audit:
    def __init__(self, m):
        self.m, self.lh, self.gh = m, [], []
    def check(self, l):
        self.lh.append(l)
        gn = np.sqrt(sum(np.sum(p.g**2) for p in self.m.ps))
        self.gh.append(gn)
        if len(self.lh) < 10: return "WARM"
        rl, rg = self.lh[-10:], self.gh[-10:]
        gs = "S"
        if np.std(rg)/(np.mean(rg)+1e-8) > 0.5: gs = "R"
        elif np.mean(rg) < 1e-4: gs = "B"
        sl = np.polyfit(np.arange(10), rl, 1)[0]
        js = "R" if sl > 0 else "B" if abs(sl) < 1e-6 else "S"
        if "R" in gs+js:
            self.m.opt.lr *= 0.8
            return f"DWN({gs}{js})"
        if gs+js == "BB":
            self.m.opt.lr = min(self.m.opt.lr*1.05, 1e-2)
            return f"UP({gs}{js})"
        return "OK"

class Model:
    def __init__(self, id, hd, od, b=4):
        self.st, self.bl = L(id, hd), [Res(hd) for _ in range(b)]
        self.hln, self.hd = LN(hd), L(hd, od)
        self.ps = self.get_ps()
        self.opt = Opt(self.ps, lr=2e-3, wd=0.05)
        self.au = Audit(self)
    def get_ps(self):
        p = self.st.get_p()
        for b in self.bl: p.extend(b.get_p())
        return p + self.hln.get_p() + self.hd.get_p()
    def forward(self, x, tr=True):
        x = self.st.forward(x)
        for b in self.bl: b.tr = tr; x = b.forward(x)
        return self.hd.forward(self.hln.forward(x))
    def backward(self, g):
        g = self.hln.backward(self.hd.backward(g))
        for b in reversed(self.bl): g = b.backward(g)
        self.st.backward(g)
    def step(self, x, y):
        lts = self.forward(x)
        ex = np.exp(lts - np.max(lts, 1, keepdims=True))
        pr = ex / ex.sum(1, keepdims=True)
        ls = -np.mean(np.sum(y * np.log(pr + 1e-12), 1))
        self.backward((pr - y) / y.shape[0])
        self.opt.step()
        return ls
    def fit(self, x, y, ep=100, bs=512):
        for e in range(1, ep+1):
            idx = np.random.permutation(len(x))
            ls = [self.step(x[idx[i:i+bs]], y[idx[i:i+bs]]) for i in range(0, len(x), bs)]
            avg = np.mean(ls)
            st = self.au.check(avg)
            ac = np.mean(np.argmax(self.forward(x[:1000], 0), 1) == np.argmax(y[:1000], 1))
            print(f"E{e:02} L:{avg:.4f} A:{ac:.3f} LR:{self.opt.lr:.5f} {st}")
            if ac > .999: break

def data(n=10000, d=784, c=10):
    x = np.random.randn(n, d).astype(np.float32)
    w = np.random.randn(d, c)
    y = np.eye(c)[np.argmax(x @ w + 0.1*np.random.randn(n, c), 1)].astype(np.float32)
    return (x - x.mean())/x.std(), y

if __name__ == "__main__":
    X, Y = data()
    m = Model(784, 128, 10, 2)
    m.fit(X, Y)