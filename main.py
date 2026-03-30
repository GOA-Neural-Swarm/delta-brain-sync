import numpy as np

class O:
    def __init__(self, p):
        self.p, self.t = p, 0
        self.m = [np.zeros_like(x) for x in p]
        self.v = [np.zeros_like(x) for x in p]
    def s(self, g):
        self.t += 1
        r = 2e-3 * (1-.999**self.t)**.5 / (1-.9**self.t)
        for i, p in enumerate(self.p):
            self.m[i] = .9*self.m[i] + .1*g[i]
            self.v[i] = .999*self.v[i] + .001*g[i]**2
            p *= .99998
            p -= r * self.m[i] / (np.sqrt(self.v[i]) + 1e-8)

class N:
    def __init__(self, d):
        self.w, self.b = np.ones((1, d), 'f4'), np.zeros((1, d), 'f4')
    def f(self, x):
        self.h = (x - (m := x.mean(-1, keepdims=1))) / (s := np.sqrt(x.var(-1, keepdims=1) + 1e-5))
        self.s = s
        return self.w * self.h + self.b
    def bk(self, d):
        dx = d * self.w
        self.dw, self.db = (d * self.h).sum(0, keepdims=1), d.sum(0, keepdims=1)
        return (dx - dx.mean(-1, keepdims=1) - self.h * (dx * self.h).mean(-1, keepdims=1)) / self.s

class S:
    def f(self, x):
        self.x, self.s = x, 1 / (1 + np.exp(-np.clip(x, -20, 20)))
        return x * self.s
    def bk(self, d): return d * (self.s + self.x * self.s * (1 - self.s))

class L:
    def __init__(self, i, o):
        self.w = np.random.randn(i, o).astype('f4') * (2/i)**.5
        self.b = np.zeros((1, o), 'f4')
    def f(self, x):
        self.x = x
        return x @ self.w + self.b
    def bk(self, d):
        self.dw, self.db = self.x.T @ d, d.sum(0, keepdims=1)
        return d @ self.w.T

class B:
    def __init__(self, d):
        self.l = [N(d), L(d, d), S(), L(d, d)]
    def f(self, x):
        h = x
        for l in self.l: h = l.f(h)
        return h + x
    def bk(self, d):
        h = d
        for l in self.l[::-1]: h = l.bk(h)
        return h + d

class M:
    def __init__(self, i=784, h=128, o=10, n=3):
        self.n = [L(i, h)] + [B(h) for _ in range(n)] + [L(h, o)]
        self.layers = [ly for l in self.n for ly in (getattr(l, 'l', [l])) if hasattr(ly, 'b')]
        self.p = [p for l in self.layers for p in (l.w, l.b)]
        self.u = O(self.p)
    def f(self, x):
        for l in self.n: x = l.f(x)
        return x
    def bk(self, d):
        for l in self.n[::-1]: d = l.bk(d)
        g = [g for l in self.layers for g in (l.dw, l.db)]
        self.u.s(g)

def step(m, x, y):
    z = m.f(x)
    p = (ex := np.exp(z - z.max(1, keepdims=1))) / ex.sum(1, keepdims=1)
    idx = np.arange(len(y))
    loss = -np.log(p[idx, y] + 1e-10).mean()
    dl = p.copy(); dl[idx, y] -= 1
    m.bk(dl / len(y))
    return loss

def train():
    X, Y = np.random.randn(100, 784).astype('f4'), np.random.randint(0, 10, 100)
    m = M()
    for e in range(101):
        for name in ["Gemini", "Groq"]:
            l = step(m, X, Y)
            if e % 10 == 0:
                acc = (m.f(X).argmax(1) == Y).mean()
                print(f"E:{e:03}|{name[0]}|L:{l:.4f}|A:{acc:.4f}")

if __name__ == "__main__":
    train()