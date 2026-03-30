import numpy as np
f32=np.float32

class AdamW:
    def __init__(self, p, lr=1e-3, b=(.9, .999), e=1e-8, wd=.01):
        self.p, self.lr, self.b1, self.b2, self.e, self.wd, self.t = p, lr, b[0], b[1], e, wd, 0
        self.m, self.v = [np.zeros_like(x) for x in p], [np.zeros_like(x) for x in p]
    def step(self, g):
        self.t += 1
        at = self.lr * (1 - self.b2**self.t)**.5 / (1 - self.b1**self.t)
        for i, p in enumerate(self.p):
            p *= (1 - self.lr * self.wd)
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g[i]
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g[i]**2
            p -= at * self.m[i] / (np.sqrt(self.v[i]) + self.e)

class LN:
    def __init__(self, d): self.g, self.b, self.e = np.ones((1, d), f32), np.zeros((1, d), f32), 1e-5
    def forward(self, x):
        self.xh = (x - (u := x.mean(-1, keepdims=1))) / np.sqrt((self.s := x.var(-1, keepdims=1)) + self.e)
        return self.g * self.xh + self.b
    def backward(self, d):
        self.dg, self.db = (d * self.xh).sum(0, keepdims=1), d.sum(0, keepdims=1)
        dxh = d * self.g
        return (dxh - dxh.mean(-1, keepdims=1) - self.xh * (dxh * self.xh).mean(-1, keepdims=1)) / np.sqrt(self.s + self.e)

class Swish:
    def forward(self, x): self.x, self.s = x, 1 / (1 + np.exp(-np.clip(x, -20, 20))); return x * self.s
    def backward(self, d): return d * (self.s + self.x * self.s * (1 - self.s))

class Linear:
    def __init__(self, i, o): self.w, self.b = np.random.randn(i, o).astype(f32) * (2/i)**.5, np.zeros((1, o), f32)
    def forward(self, x): self.x = x; return x @ self.w + self.b
    def backward(self, d): self.dw, self.db = self.x.T @ d, d.sum(0, keepdims=1); return d @ self.w.T

class Block:
    def __init__(self, d): self.l = [LN(d), Linear(d, d), Swish(), Linear(d, d)]
    def forward(self, x):
        h = x
        for l in self.l: h = l.forward(h)
        return h + x
    def backward(self, d):
        dh = d
        for l in reversed(self.l): dh = l.backward(dh)
        return dh + d

class Engine:
    def __init__(self, i=784, h=128, o=10, n=3):
        self.net = [Linear(i, h)] + [Block(h) for _ in range(n)] + [Linear(h, o)]
        self.o = []
        for l in self.net: self.o += [x for x in getattr(l, 'l', []) if hasattr(x, 'b')] + ([l] if hasattr(l, 'b') and not hasattr(l, 'l') else [])
        self.p = [(o.w if hasattr(o, 'w') else o.g) for o in self.o] + [o.b for o in self.o]
        self.opt = AdamW(self.p, 2e-3)
    def forward(self, x):
        for l in self.net: x = l.forward(x)
        return x
    def backward(self, d):
        for l in reversed(self.net): d = l.backward(d)
        g = [(o.dw if hasattr(o, 'dw') else o.dg) for o in self.o] + [o.db for o in self.o]
        self.opt.step(g)

def train():
    X, Y, m, r = np.random.randn(100, 784).astype(f32), np.random.randint(0, 10, 100), Engine(), range(100)
    for e in range(101):
        z = m.forward(X)
        p = (ex := np.exp(z - z.max(1, keepdims=1))) / ex.sum(1, keepdims=1)
        loss = -np.log(p[r, Y] + 1e-10).mean()
        dl = p.copy(); dl[r, Y] -= 1
        m.backward(dl / 100)
        if e % 10 == 0: print(f"E:{e:03} | L:{loss:.4f} | A:{(p.argmax(1)==Y).mean():.4f}")

if __name__ == "__main__":
    train()