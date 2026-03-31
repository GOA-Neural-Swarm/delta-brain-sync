import numpy as np

class Normalization:
    def __init__(self, d):
        self.g = np.ones(d)
        self.bt = np.zeros(d)

    def __call__(self, x):
        self.n = (x - x.mean(0)) / ((x.var(0) + 1e-5) ** 0.5)
        return self.n * self.g + self.bt

    def backward(self, d):
        self.dg = (d * self.n).sum(0)
        self.dbt = d.sum(0)
        return d * self.g

class Activation:
    def __call__(self, x):
        self.o = 1 / (1 + np.exp(-np.clip(x, -20, 20)))
        return self.o

    def backward(self, d):
        return d * self.o * (1 - self.o)

class Linear:
    def __init__(self, i, o):
        self.w = np.random.randn(i, o) * (2 / i) ** 0.5
        self.b = np.zeros(o)

    def __call__(self, x):
        self.x = x
        return x @ self.w + self.b

    def backward(self, d):
        self.dw = self.x.T @ d
        self.db = d.sum(0)
        return d @ self.w.T

class Block:
    def __init__(self, d, f):
        self.l = [Normalization(d), Linear(d, d * f), Activation(), Linear(d * f, d)]

    def __call__(self, x):
        h = x
        for l in self.l:
            h = l(h)
        return h + x

    def backward(self, d):
        g = d
        for l in reversed(self.l):
            g = l.backward(g)
        return d + g

class Model:
    def __init__(self, i=784, h=128, o=10, c=3):
        self.l = [Linear(i, h)] + [Block(h, 4) for _ in range(c)] + [Block(h, 2), Block(h, 3), Linear(h, o)]
        self.p = [(s, a) for l in self.l for s in (l.l if hasattr(l, 'l') else [l]) for a in ('w', 'g', 'b', 'bt') if hasattr(s, a)]
        self.m = [np.zeros(getattr(s, a).shape) for s, a in self.p]
        self.v = [0 * x for x in self.m]
        self.t = 0

    def forward(self, x):
        for l in self.l:
            x = l(x)
        return x

    def backward(self, d):
        for l in reversed(self.l):
            d = l.backward(d)
        self.t += 1
        r = 2e-3 * (1 - 0.999 ** self.t) ** 0.5 / (1 - 0.9 ** self.t)
        for i, (s, a) in enumerate(self.p):
            g = getattr(s, 'd' + a) if hasattr(s, 'd' + a) else getattr(s, a)
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g ** 2
            setattr(s, a, getattr(s, a) - r * self.m[i] / (self.v[i] ** 0.5 + 1e-8))

X = np.random.randn(100, 784).astype('f4')
Y = np.random.randint(10, size=100)
I = np.arange(100)

m = Model()

for e in range(101):
    z = m.forward(X)
    v = np.exp(z - z.max(1, keepdims=1))
    p = v / v.sum(1, keepdims=1)
    l = -np.log(p[np.arange(100), Y] + 1e-9).mean()
    d = p.copy()
    d[np.arange(100), Y] -= 1
    m.backward(d)
    if e % 10 == 0:
        print(f"E:{e} L:{l:.2f} A:{(z.argmax(1) == Y).mean():.2f}")