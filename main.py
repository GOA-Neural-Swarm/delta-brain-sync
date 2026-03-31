import numpy as np

R = np.random.randn

class N:
    def __init__(self, d): 
        self.g, self.b = np.ones(d), np.zeros(d)
        self.dg, self.db = np.zeros(d), np.zeros(d)
    def __call__(self, x):
        self.n = (x - x.mean(0)) / (x.var(0) + 1e-5)**.5
        return self.n * self.g + self.b
    def backward(self, d):
        self.dg = (d * self.n).sum(0)
        self.db = d.sum(0)
        return d * self.g

class A:
    def __call__(self, x): 
        self.o = 1 / (1 + np.exp(-x.clip(-20, 20)))
        return self.o
    def backward(self, d): 
        return d * self.o * (1 - self.o)

class L:
    def __init__(self, i, o): 
        self.w, self.b = R(i, o) * (2 / i)**.5, np.zeros(o)
        self.dw, self.db = np.zeros((i, o)), np.zeros(o)
        self.x = None
    def __call__(self, x): 
        self.x = x
        return x @ self.w + self.b
    def backward(self, d):
        self.dw = self.x.T @ d
        self.db = d.sum(0)
        return d @ self.w.T

class B:
    def __init__(self, d, f): 
        self.l = [N(d), L(d, d * f), A(), L(d * f, d)]
    def __call__(self, x):
        h = x
        for l in self.l: h = l(h)
        return h + x
    def backward(self, d):
        g = d
        for l in reversed(self.l): g = l.backward(g)
        return d + g

class M:
    def __init__(self, i=784, h=128, o=10, c=3):
        self.L = [L(i, h)] + [B(h, 4) for _ in range(c)] + [B(h, 2), B(h, 3), L(h, o)]
        self.p = [(s, a) for l in self.L for s in (l.l if hasattr(l, 'l') else [l]) for a in ('w', 'g', 'b') if hasattr(s, a)]
        self.m = [np.zeros(getattr(s, a).shape) for s, a in self.p]
        self.v = [x * 0 for x in self.m]
        self.t = 0
    def fwd(self, x):
        for l in self.L: x = l(x)
        return x
    def bwd(self, d):
        for l in reversed(self.L): d = l.backward(d)
        self.t += 1
        r = 2e-3 * (1 - .999**self.t)**.5 / (1 - .9**self.t)
        for i, (s, a) in enumerate(self.p):
            g = getattr(s, 'd' + a) if hasattr(s, 'd' + a) else getattr(s, a)
            self.m[i] = .9 * self.m[i] + .1 * g
            self.v[i] = .999 * self.v[i] + .001 * g**2
            if a == 'w':
                setattr(s, a, getattr(s, a) - r * self.m[i] / (self.v[i]**.5 + 1e-8))
            elif a == 'g' or a == 'b':
                setattr(s, a, getattr(s, a) - r * self.m[i] / (self.v[i]**.5 + 1e-8))

X, Y, S = R(100, 784).astype('f4'), np.random.randint(0, 10, 100), 100
m = M()

for e in range(101):
    z = m.fwd(X)
    v = np.exp(z - z.max(1, keepdims=True))
    p = v / v.sum(1, keepdims=True)
    loss = -np.log(p[range(S), Y] + 1e-9).mean()
    d = p.copy()
    d[range(S), Y] -= 1
    m.bwd(d / S)
    if e % 10 == 0:
        print(f"E:{e} L:{loss:.2f} A:{(z.argmax(1) == Y).mean():.2f}")