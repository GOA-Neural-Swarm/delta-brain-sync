import numpy as np

class N:
    def __init__(self, d): self.w, self.b = np.ones(d), np.zeros(d)
    def f(self, x):
        self.m, self.v = x.mean(-1, keepdims=1), x.var(-1, keepdims=1)
        self.h = (x - self.m) / (self.v + 1e-5)**.5
        return self.w * self.h + self.b
    def b(self, d):
        dx = d * self.w
        self.dw, self.db = (d * self.h).sum(0), d.sum(0)
        return (dx - dx.mean(-1, keepdims=1) - self.h * (dx * self.h).mean(-1, keepdims=1)) / (self.v + 1e-5)**.5

class A:
    def f(self, x):
        self.s = 1 / (1 + np.exp(-np.clip(x, -20, 20)))
        self.x = x
        return x * self.s
    def b(self, d): return d * (self.s + self.x * self.s * (1 - self.s))

class L:
    def __init__(self, i, o): self.w, self.b = np.random.randn(i, o) * (2/i)**.5, np.zeros(o)
    def f(self, x):
        self.x = x
        return x @ self.w + self.b
    def b(self, d):
        self.dw, self.db = self.x.T @ d, d.sum(0)
        return d @ self.w.T

class B:
    def __init__(self, d): self.l = [N(d), L(d, d), A(), L(d, d)]
    def f(self, x):
        h = x
        for l in self.l: h = l.f(h)
        return h + x
    def b(self, d):
        g = d
        for l in reversed(self.l): g = l.b(g)
        return d + g

class M:
    def __init__(self, i=784, h=128, o=10, c=3):
        self.ls = [L(i, h)] + [B(h) for _ in range(c)] + [L(h, o)]
        self.p, self.t = [], 0
        for l in self.ls:
            for s in (l.l if hasattr(l, 'l') else [l]):
                if hasattr(s, 'w'): self.p.append(s)
        self.m = [np.zeros_like(getattr(p, a)) for p in self.p for a in ('w', 'b')]
        self.v = [np.zeros_like(x) for x in self.m]
    def f(self, x):
        for l in self.ls: x = l.f(x)
        return x
    def b(self, d, u):
        for l in reversed(self.ls): d = l.b(d)
        self.t += 1
        r = 2e-3 * (1 - .999**self.t)**.5 / (1 - .9**self.t)
        for i, p in enumerate(self.p):
            for j, a in enumerate(['w', 'b']):
                idx, g = i*2 + j, getattr(p, 'd'+a) / u
                self.m[idx] = .9 * self.m[idx] + .1 * g
                self.v[idx] = .999 * self.v[idx] + .001 * g**2
                getattr(p, a)[:] -= r * self.m[idx] / (self.v[idx]**.5 + 1e-8)

X, Y, u = np.random.randn(100, 784).astype('f4'), np.random.randint(10, size=100), 100
m = M()
for e in range(101):
    z = m.f(X)
    p = (v := np.exp(z - z.max(1, keepdims=1))) / v.sum(1, keepdims=1)
    ls = -np.log(p[range(100), Y] + 1e-9).mean()
    d = p.copy(); d[range(100), Y] -= 1
    m.b(d, u)
    if e % 10 == 0: print(f"E:{e} L:{ls:.2f} A:{(z.argmax(1)==Y).mean():.2f}")