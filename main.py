import numpy as np

class Optimizer:
    def __init__(self, params):
        self.params = params
        self.t = 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
    def step(self, grads):
        self.t += 1
        r = 2e-3 * (1-.999**self.t)**.5 / (1-.9**self.t)
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = .9*self.m[i] + .1*g
            self.v[i] = .999*self.v[i] + .001*g**2
            p *= .99998
            p -= r * self.m[i] / (np.sqrt(self.v[i]) + 1e-8)

class Normalization:
    def __init__(self, dim):
        self.w, self.b = np.ones((1, dim), 'f4'), np.zeros((1, dim), 'f4')
    def forward(self, x):
        self.h = (x - (m := x.mean(-1, keepdims=1))) / (s := np.sqrt(x.var(-1, keepdims=1) + 1e-5))
        self.s = s
        return self.w * self.h + self.b
    def backward(self, d):
        dx = d * self.w
        self.dw, self.db = (d * self.h).sum(0, keepdims=1), d.sum(0, keepdims=1)
        return (dx - dx.mean(-1, keepdims=1) - self.h * (dx * self.h).mean(-1, keepdims=1)) / self.s

class Sigmoid:
    def forward(self, x):
        self.x, self.s = x, 1 / (1 + np.exp(-np.clip(x, -20, 20)))
        return x * self.s
    def backward(self, d): return d * (self.s + self.x * self.s * (1 - self.s))

class Linear:
    def __init__(self, i, o):
        self.w = np.random.randn(i, o).astype('f4') * (2/i)**.5
        self.b = np.zeros((1, o), 'f4')
    def forward(self, x):
        self.x = x
        return x @ self.w + self.b
    def backward(self, d):
        self.dw, self.db = self.x.T @ d, d.sum(0, keepdims=1)
        return d @ self.w.T

class Block:
    def __init__(self, dim):
        self.layers = [Normalization(dim), Linear(dim, dim), Sigmoid(), Linear(dim, dim)]
    def forward(self, x):
        h = x
        for l in self.layers: h = l.forward(h)
        return h + x
    def backward(self, d):
        h = d
        for l in self.layers[::-1]: h = l.backward(h)
        return h + d

class Model:
    def __init__(self, i=784, h=128, o=10, n=3):
        self.blocks = [Linear(i, h)] + [Block(h) for _ in range(n)] + [Linear(h, o)]
        self.layers = [ly for l in self.blocks for ly in (getattr(l, 'layers', [l])) if hasattr(ly, 'b')]
        self.params = [p for l in self.layers for p in (l.w, l.b)]
        self.optimizer = Optimizer(self.params)
    def forward(self, x):
        for l in self.blocks: x = l.forward(x)
        return x
    def backward(self, d):
        for l in self.blocks[::-1]: d = l.backward(d)
        grads = [g for l in self.layers for g in (l.dw, l.db)]
        self.optimizer.step(grads)

def step(model, x, y):
    z = model.forward(x)
    p = (ex := np.exp(z - z.max(1, keepdims=1))) / ex.sum(1, keepdims=1)
    idx = np.arange(len(y))
    loss = -np.log(p[idx, y] + 1e-10).mean()
    dl = p.copy(); dl[idx, y] -= 1
    model.backward(dl / len(y))
    return loss

def train():
    X, Y = np.random.randn(100, 784).astype('f4'), np.random.randint(0, 10, 100)
    model = Model()
    for e in range(101):
        l = step(model, X, Y)
        if e % 10 == 0:
            acc = (model.forward(X).argmax(1) == Y).mean()
            print(f"E:{e:03}|L:{l:.4f}|A:{acc:.4f}")

if __name__ == "__main__":
    train()