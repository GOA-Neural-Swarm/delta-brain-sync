import numpy as np

class Normalization:
    def __init__(self, d):
        self.gamma = np.ones(d)
        self.beta = np.zeros(d)

    def __call__(self, x):
        self.normalized = (x - x.mean(0)) / np.sqrt(x.var(0) + 1e-5)
        return self.normalized * self.gamma + self.beta

    def back(self, d):
        self.dgamma = (d * self.normalized).sum(0)
        self.dbeta = d.sum(0)
        return d * self.gamma

class Activation:
    def __call__(self, x):
        self.output = 1 / (1 + np.exp(-np.clip(x, -20, 20)))
        return self.output

    def back(self, d):
        return d * self.output * (1 - self.output)

class Linear:
    def __init__(self, i, o):
        self.weight = np.random.randn(i, o) * (2/i)**0.5
        self.bias = np.zeros(o)

    def __call__(self, x):
        self.x = x
        return x @ self.weight + self.bias

    def back(self, d):
        self.dweight = self.x.T @ d
        self.dbias = d.sum(0)
        return d @ self.weight.T

class Block:
    def __init__(self, d, f):
        self.layers = [Normalization(d), Linear(d, d*f), Activation(), Linear(d*f, d)]

    def __call__(self, x):
        h = x
        for l in self.layers:
            h = l(h)
        return h + x

    def back(self, d):
        g = d
        for l in reversed(self.layers):
            g = l.back(g)
        return d + g

class Model:
    def __init__(self, i=784, h=128, o=10, c=3):
        self.layers = [Linear(i, h)] + [Block(h, 4) for _ in range(c)] + [Block(h, 2), Block(h, 3), Linear(h, o)]
        self.params, self.t = [], 0
        for l in self.layers:
            if hasattr(l, 'layers'):
                for s in l.layers:
                    for a in ('weight', 'gamma', 'bias'):
                        if hasattr(s, a):
                            self.params.append((s, a))
            else:
                for a in ('weight', 'gamma', 'bias'):
                    if hasattr(l, a):
                        self.params.append((l, a))
        self.m = [np.zeros_like(getattr(s, a)) for s, a in self.params]
        self.v = [np.zeros_like(x) for x in self.m]

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def backward(self, d):
        for l in reversed(self.layers):
            d = l.back(d)
        self.t += 1
        r = 2e-3 * (1 - 0.999**self.t)**0.5 / (1 - 0.9**self.t)
        for i, (s, a) in enumerate(self.params):
            g = getattr(s, 'd' + a if a != 'gamma' else 'dgamma')
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g**2
            setattr(s, a, getattr(s, a) - r * self.m[i] / (np.sqrt(self.v[i]) + 1e-8))

X, Y = np.random.randn(100, 784).astype('f4'), np.random.randint(10, size=100)
m = Model()
for e in range(101):
    z = m.forward(X)
    p = (v := np.exp(z - z.max(1, keepdims=1))) / v.sum(1, keepdims=1)
    loss = -np.log(p[range(100), Y] + 1e-9).mean()
    d = p.copy()
    d[range(100), Y] -= 1
    m.backward(d)
    if e % 10 == 0:
        print(f"E:{e} L:{loss:.2f} A:{(z.argmax(1)==Y).mean():.2f}")