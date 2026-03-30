import numpy as np

class AdamW:
    def __init__(self, p, lr=1e-3, b=(.9, .999), e=1e-8, wd=.01):
        self.p, self.lr, self.b1, self.b2, self.e, self.wd, self.t = p, lr, b[0], b[1], e, wd, 0
        self.m = [np.zeros_like(x) for x in p]
        self.v = [np.zeros_like(x) for x in p]

    def step(self, g):
        self.t += 1
        at = self.lr * (1 - self.b2**self.t)**.5 / (1 - self.b1**self.t)
        for i in range(len(self.p)):
            self.p[i] -= self.lr * self.wd * self.p[i]
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g[i]
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g[i]**2
            self.p[i] -= at * self.m[i] / (np.sqrt(self.v[i]) + self.e)

class LN:
    def __init__(self, d):
        self.g, self.b, self.e = np.ones((1, d)), np.zeros((1, d)), 1e-5

    def forward(self, x):
        self.x = x
        self.u = x.mean(-1, keepdims=1)
        self.s = x.var(-1, keepdims=1)
        self.xh = (x - self.u) / np.sqrt(self.s + self.e)
        return self.g * self.xh + self.b

    def backward(self, d):
        self.dg, self.db = (d * self.xh).sum(0, 1), d.sum(0, 1)
        dxh = d * self.g
        D = d.shape[-1]
        return (dxh - dxh.mean(-1, 1) - self.xh * (dxh * self.xh).mean(-1, 1)) / np.sqrt(self.s + self.e)

class Swish:
    def forward(self, x):
        self.x = x
        self.sig = 1 / (1 + np.exp(-np.clip(x, -20, 20)))
        return x * self.sig
    def backward(self, d):
        return d * (self.sig + self.x * self.sig * (1 - self.sig))

class Linear:
    def __init__(self, i, o):
        self.w = np.random.randn(i, o).astype(np.float32) * (2/i)**.5
        self.b = np.zeros((1, o), dtype=np.float32)
    def forward(self, x):
        self.x = x
        return x @ self.w + self.b
    def backward(self, d):
        self.dw, self.db = self.x.T @ d, d.sum(0, 1)
        return d @ self.w.T

class Block:
    def __init__(self, d):
        self.layers = [LN(d), Linear(d, d), Swish(), Linear(d, d)]
    def forward(self, x):
        h = x
        for l in self.layers: h = l.forward(h)
        return h + x
    def backward(self, d):
        dh = d
        for l in reversed(self.layers): dh = l.backward(dh)
        return dh + d

class Gemini:
    def __init__(self, d):
        self.layers = [LN(d), Linear(d, d), Swish(), Linear(d, d)]
    def forward(self, x):
        h = x
        for l in self.layers: h = l.forward(h)
        return h + x
    def backward(self, d):
        dh = d
        for l in reversed(self.layers): dh = l.backward(dh)
        return dh + d

class Groq:
    def __init__(self, d):
        self.layers = [LN(d), Linear(d, d), Swish(), Linear(d, d)]
    def forward(self, x):
        h = x
        for l in self.layers: h = l.forward(h)
        return h + x
    def backward(self, d):
        dh = d
        for l in reversed(self.layers): dh = l.backward(dh)
        return dh + d

class Engine:
    def __init__(self, i=784, h=256, o=10, n=3):
        self.net = [Linear(i, h)] + [Block(h) for _ in range(n)] + [Linear(h, o)]
        self.gemini = Gemini(h)
        self.groq = Groq(h)
        self.flat = []
        for l in self.net:
            self.flat.extend(l.layers if hasattr(l, 'layers') else [l])
        self.flat.extend(self.gemini.layers)
        self.flat.extend(self.groq.layers)
        self.p = []
        for l in self.flat:
            if hasattr(l, 'w'): self.p += [l.w, l.b]
            if hasattr(l, 'g'): self.p += [l.g, l.b]
        self.opt = AdamW(self.p, 2e-3)

    def forward(self, x):
        for l in self.net: x = l.forward(x)
        x = self.gemini.forward(x)
        x = self.groq.forward(x)
        return x

    def backward(self, d):
        d = self.groq.backward(d)
        d = self.gemini.backward(d)
        for l in reversed(self.net): d = l.backward(d)
        g = []
        for l in self.flat:
            if hasattr(l, 'dw'): g += [l.dw, l.db]
            if hasattr(l, 'dg'): g += [l.dg, l.db]
        self.opt.step(g)

def train():
    X, Y = np.random.randn(100, 784).astype(np.float32), np.random.randint(0, 10, 100)
    m = Engine(784, 128, 10, 3)
    for e in range(101):
        logits = m.forward(X)
        ex = np.exp(logits - logits.max(1, keepdims=1))
        p = ex / ex.sum(1, keepdims=1)
        loss = -np.mean(np.log(p[range(100), Y] + 1e-10))
        dl = p.copy()
        dl[range(100), Y] -= 1
        m.backward(dl / 100)
        if e % 10 == 0:
            acc = (p.argmax(1) == Y).mean()
            print(f"E:{e:03} | L:{loss:.4f} | A:{acc:.4f}")

if __name__ == "__main__":
    train()