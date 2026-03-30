import numpy as np

class N:
    def __init__(self, d): self.g, self.b = np.ones(d), np.zeros(d)
    def __call__(self, x):
        self.n = (x - (u := x.mean(0))) / np.sqrt(x.var(0) + 1e-5)
        return self.n * self.g + self.b
    def back(self, d):
        self.dg, self.db = (d * self.n).sum(0), d.sum(0)
        return d * self.g

class A:
    def __call__(self, x): self.o = 1 / (1 + np.exp(-np.clip(x, -20, 20))); return self.o
    def back(self, d): return d * self.o * (1 - self.o)

class L:
    def __init__(self, i, o): self.w, self.b = np.random.randn(i, o) * (2/i)**0.5, np.zeros(o)
    def __call__(self, x): self.x = x; return x @ self.w + self.b
    def back(self, d):
        self.dw, self.db = self.x.T @ d, d.sum(0)
        return d @ self.w.T

class B:
    def __init__(self, d, f): self.l = [N(d), L(d, d*f), A(), L(d*f, d)]
    def __call__(self, x):
        h = x
        for l in self.l: h = l(h)
        return h + x
    def back(self, d):
        g = d
        for l in reversed(self.l): g = l.back(g)
        return d + g

class Model:
    def __init__(self, i=784, h=128, o=10, c=3):
        # Optimized sequence to ensure dimension compatibility (Gemini=f2, Groq=f3)
        self.ls = [L(i, h)] + [B(h, 4) for _ in range(c)] + [B(h, 2), B(h, 3), L(h, o)]
        self.p, self.t = [], 0
        for l in self.ls:
            for s in (l.l if hasattr(l, 'l') else [l]):
                for a in ('w', 'g', 'b'):
                    if hasattr(s, a): self.p.append((s, a))
        self.m = [np.zeros_like(getattr(s, a)) for s, a in self.p]
        self.v = [np.zeros_like(x) for x in self.m]

    def forward(self, x):
        for l in self.ls: x = l(x)
        return x

    def backward(self, d):
        for l in reversed(self.ls): d = l.back(d)
        self.t += 1
        r = 2e-3 * (1 - 0.999**self.t)**0.5 / (1 - 0.9**self.t)
        for i, (s, a) in enumerate(self.p):
            g = getattr(s, 'd'+a if a != 'g' else 'dg')
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g**2
            setattr(s, a, getattr(s, a) - r * self.m[i] / (np.sqrt(self.v[i]) + 1e-8))

X, Y = np.random.randn(100, 784).astype('f4'), np.random.randint(10, size=100)
m = Model()
for e in range(101):
    z = m.forward(X)
    p = (v := np.exp(z - z.max(1, keepdims=1))) / v.sum(1, keepdims=1)
    loss = -np.log(p[range(100), Y] + 1e-9).mean()
    d = p.copy(); d[range(100), Y] -= 1
    m.backward(d)
    if e % 10 == 0:
        print(f"E:{e} L:{loss:.2f} A:{(z.argmax(1)==Y).mean():.2f}")