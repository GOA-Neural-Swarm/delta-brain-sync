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

class Block:
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

class Gemini:
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

class Groq:
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
        self.L = [Linear(i, h)] + [Block(h, f) for f in [4]*c + [2, 3]] + [Linear(h, o)]
        self.p, self.m, self.v, self.t = [], [], [], 0
        for l in self.L:
            if isinstance(l, Block):
                for s in l.l:
                    for a in 'wgb':
                        if hasattr(s, a):
                            self.p.append((s, a))
                            self.m.append(0)
                            self.v.append(0)
            else:
                for a in 'wgb':
                    if hasattr(l, a):
                        self.p.append((l, a))
                        self.m.append(0)
                        self.v.append(0)

    def fwd(self, x):
        for l in self.L:
            x = l(x)
        return x

    def bwd(self, d):
        for l in reversed(self.L):
            d = l.bwd(d)
        self.t += 1
        lr = 2e-3 * (1 - .999**self.t)**.5 / (1 - .9**self.t)
        for i, (s, a) in enumerate(self.p):
            g = getattr(s, 'd' + a) if hasattr(s, 'd' + a) else getattr(s, a)
            self.m[i] = .9 * self.m[i] + .1 * g
            self.v[i] = .999 * self.v[i] + .001 * g**2
            setattr(s, a, getattr(s, a) - lr * self.m[i] / (self.v[i]**.5 + 1e-8))

X, Y, S = np.random.randn(100, 784).astype('f4'), np.random.randint(0, 10, 100), 100
m = Model()
for e in range(101):
    z = m.fwd(X)
    p = (v := np.exp(z - z.max(1, keepdims=1))) / v.sum(1, keepdims=1)
    ls = -np.log(p[range(S), Y] + 1e-9).mean()
    d = p.copy()
    d[range(S), Y] -= 1
    m.bwd(d / S)
    if e % 10 == 0:
        print(f"E:{e} L:{ls:.2f} A:{(z.argmax(1) == Y).mean():.2f}")