import numpy as np

class Normalization:
    def __init__(self, d):
        self.g = np.ones(d)
        self.b = np.zeros(d)

    def __call__(self, x):
        self.h = (x - x.mean(0)) / (x.var(0) + 1e-5)**.5
        return self.h * self.g + self.b

    def bwd(self, d):
        self.dg = (d * self.h).sum(0)
        self.db = d.sum(0)
        return d * self.g

class Activation:
    def __call__(self, x):
        self.o = 1 / (1 + np.exp(-np.clip(x, -20, 20)))
        return self.o

    def bwd(self, d):
        return d * self.o * (1 - self.o)

class Linear:
    def __init__(self, i, o):
        self.w = np.random.randn(i, o) * (2/i)**.5
        self.b = np.zeros(o)

    def __call__(self, x):
        self.x = x
        return x @ self.w + self.b

    def bwd(self, d):
        self.dw = self.x.T @ d
        self.db = d.sum(0)
        return d @ self.w.T

class ModularBlock:
    def __init__(self, d, f):
        self.l = [Normalization(d), Linear(d, d*f), Activation(), Linear(d*f, d)]

    def __call__(self, x):
        h = x
        for l in self.l:
            h = l(h)
        return h + x

    def bwd(self, d):
        g = d
        for l in reversed(self.l):
            g = l.bwd(g)
        return d + g

class Model:
    def __init__(self, i=784, h=128, o=10, c=3):
        self.layers = [Linear(i, h)] + [ModularBlock(h, f) for f in [4]*c] + [ModularBlock(h, 2), ModularBlock(h, 3)] + [Linear(h, o)]
        self.params, self.t = [], 0
        for l in self.layers:
            if hasattr(l, 'l'):
                for s in l.l:
                    for a in 'wgb':
                        if hasattr(s, a):
                            v = getattr(s, a)
                            self.params.append([s, a, np.zeros_like(v), np.zeros_like(v)])
            else:
                for a in 'wb':
                    if hasattr(l, a):
                        v = getattr(l, a)
                        self.params.append([l, a, np.zeros_like(v), np.zeros_like(v)])

    def fwd(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def bwd(self, d):
        for l in reversed(self.layers):
            d = l.bwd(d)
        self.t += 1
        lr = 2e-3 * (1 - .999**self.t)**.5 / (1 - .9**self.t)
        for p in self.params:
            s, a, m, v = p
            if hasattr(s, 'dw') and a == 'w':
                g = s.dw
            elif hasattr(s, 'db') and a == 'b':
                g = s.db
            elif hasattr(s, 'dg') and a == 'g':
                g = s.dg
            elif hasattr(s, 'db') and a == 'b':
                g = s.db
            m[:] = .9 * m + .1 * g
            v[:] = .999 * v + .001 * g**2
            if a == 'w':
                setattr(s, a, getattr(s, a) - lr * m / (np.sqrt(v) + 1e-8))
            elif a == 'b':
                setattr(s, a, getattr(s, a) - lr * m / (np.sqrt(v) + 1e-8))
            elif a == 'g':
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