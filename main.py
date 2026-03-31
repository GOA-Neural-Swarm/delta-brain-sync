import numpy as np

class N:
    def __init__(self, d): self.g, self.b = np.ones(d), np.zeros(d)
    def __call__(self, x):
        self.h = (x - x.mean(0)) / (x.var(0) + 1e-5)**.5
        return self.h * self.g + self.b
    def bwd(self, d):
        self.dg, self.db = (d * self.h).sum(0), d.sum(0)
        return d * self.g

class A:
    def __call__(self, x):
        self.o = 1 / (1 + np.exp(-np.clip(x, -20, 20)))
        return self.o
    def bwd(self, d): return d * self.o * (1 - self.o)

class L:
    def __init__(self, i, o):
        self.w = np.random.randn(i, o) * (2/i)**.5
        self.b = np.zeros(o)
    def __call__(self, x):
        self.x = x
        return x @ self.w + self.b
    def bwd(self, d):
        self.dw, self.db = self.x.T @ d, d.sum(0)
        return d @ self.w.T

class B:
    def __init__(self, d, f):
        self.l = [N(d), L(d, d*f), A(), L(d*f, d)]
    def __call__(self, x):
        h = x
        for l in self.l: h = l(h)
        return h + x
    def bwd(self, d):
        g = d
        for l in reversed(self.l): g = l.bwd(g)
        return d + g

class Model:
    def __init__(self, i=784, h=128, o=10, c=3):
        self.L = [L(i, h)] + [B(h, f) for f in [4]*c + [2, 3]] + [L(h, o)]
        self.p, self.t = [], 0
        for l in self.L:
            for s in (l.l if hasattr(l, 'l') else [l]):
                for a in 'wgb':
                    if hasattr(s, a):
                        v = getattr(s, a)
                        self.p.append([s, a, np.zeros_like(v), np.zeros_like(v)])

    def fwd(self, x):
        for l in self.L: x = l(x)
        return x

    def bwd(self, d):
        for l in reversed(self.L): d = l.bwd(d)
        self.t += 1
        lr = 2e-3 * (1 - .999**self.t)**.5 / (1 - .9**self.t)
        for p in self.p:
            s, a, m, v = p
            g = getattr(s, 'd' + a)
            m[:] = .9 * m + .1 * g
            v[:] = .999 * v + .001 * g**2
            setattr(s, a, getattr(s, a) - lr * m / (np.sqrt(v) + 1e-8))

X, Y, S = np.random.randn(100, 784).astype('f4'), np.random.randint(0, 10, 100), 100
m = Model()
for e in range(101):
    z = m.fwd(X)
    p = (exp := np.exp(z - z.max(1, keepdims=1))) / exp.sum(1, keepdims=1)
    ls = -np.log(p[range(S), Y] + 1e-9).mean()
    d = p.copy(); d[range(S), Y] -= 1
    m.bwd(d / S)
    if e % 10 == 0:
        print(f"E:{e} L:{ls:.2f} A:{(z.argmax(1) == Y).mean():.2f}")