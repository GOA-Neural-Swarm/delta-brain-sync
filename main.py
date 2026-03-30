import numpy as np

class Normalization:
    def __init__(self, d):
        self.w, self.b = np.ones(d), np.zeros(d)
        self.moving_mean, self.moving_var = np.zeros(d), np.ones(d)
        self.gamma, self.beta = np.ones(d), np.zeros(d)
    def forward(self, x):
        self.batch_mean, self.batch_var = x.mean(0), x.var(0)
        self.moving_mean = 0.9 * self.moving_mean + 0.1 * self.batch_mean
        self.moving_var = 0.9 * self.moving_var + 0.1 * self.batch_var
        self.h = (x - self.batch_mean) / np.sqrt(self.batch_var + 1e-5)
        return self.h * self.gamma + self.beta
    def backward(self, d):
        dx = d * self.gamma
        self.dgamma = (d * self.h).sum(0)
        self.dbeta = d.sum(0)
        return dx

class Activation:
    def forward(self, x):
        self.s = 1 / (1 + np.exp(-np.clip(x, -20, 20)))
        return self.s
    def backward(self, d):
        return d * self.s * (1 - self.s)

class Linear:
    def __init__(self, i, o):
        self.w, self.b = np.random.randn(i, o) * (2/i)**0.5, np.zeros(o)
    def forward(self, x):
        self.x = x
        return x @ self.w + self.b
    def backward(self, d):
        self.dw, self.db = self.x.T @ d, d.sum(0)
        return d @ self.w.T

class Bottleneck:
    def __init__(self, d):
        self.l = [Normalization(d), Linear(d, d*4), Activation(), Linear(d*4, d)]
    def forward(self, x):
        h = x
        for l in self.l: h = l.forward(h)
        return h + x
    def backward(self, d):
        g = d
        for l in reversed(self.l): g = l.backward(g)
        return d + g

class Gemini:
    def __init__(self, d):
        self.l = [Normalization(d), Linear(d, d*2), Activation(), Linear(d*2, d)]
    def forward(self, x):
        h = x
        for l in self.l: h = l.forward(h)
        return h + x
    def backward(self, d):
        g = d
        for l in reversed(self.l): g = l.backward(g)
        return d + g

class Groq:
    def __init__(self, d):
        self.l = [Normalization(d), Linear(d, d*3), Activation(), Linear(d*3, d)]
    def forward(self, x):
        h = x
        for l in self.l: h = l.forward(h)
        return h + x
    def backward(self, d):
        g = d
        for l in reversed(self.l): g = l.backward(g)
        return d + g

class Model:
    def __init__(self, i=784, h=128, o=10, c=3):
        self.ls = [Linear(i, h)] + [Bottleneck(h) for _ in range(c)] + [Linear(h, o)]
        self.gemini = Gemini(h)
        self.groq = Groq(h)
        self.p, self.t = [], 0
        for l in self.ls:
            if hasattr(l, 'l'):
                for s in l.l:
                    if hasattr(s, 'w'): self.p.append(s)
            else:
                if hasattr(l, 'w'): self.p.append(l)
        self.m = [np.zeros_like(getattr(p, a)) for p in self.p for a in ('w', 'b', 'gamma', 'beta')]
        self.v = [np.zeros_like(x) for x in self.m]
    def forward(self, x):
        for l in self.ls: x = l.forward(x)
        x = self.gemini.forward(x)
        x = self.groq.forward(x)
        return x
    def backward(self, d, u):
        d = self.groq.backward(d)
        d = self.gemini.backward(d)
        for l in reversed(self.ls): d = l.backward(d)
        self.t += 1
        r = 2e-3 * (1 - 0.999**self.t)**0.5 / (1 - 0.9**self.t)
        for i, p in enumerate(self.p):
            if hasattr(p, 'gamma') and hasattr(p, 'beta'):
                self.m[i*4] = 0.9 * self.m[i*4] + 0.1 * p.dgamma
                self.v[i*4] = 0.999 * self.v[i*4] + 0.001 * p.dgamma**2
                p.gamma -= r * self.m[i*4] / (self.v[i*4]**0.5 + 1e-8)
                self.m[i*4+1] = 0.9 * self.m[i*4+1] + 0.1 * p.dbeta
                self.v[i*4+1] = 0.999 * self.v[i*4+1] + 0.001 * p.dbeta**2
                p.beta -= r * self.m[i*4+1] / (self.v[i*4+1]**0.5 + 1e-8)
            self.m[i*4+2] = 0.9 * self.m[i*4+2] + 0.1 * p.dw
            self.v[i*4+2] = 0.999 * self.v[i*4+2] + 0.001 * p.dw**2
            p.w -= r * self.m[i*4+2] / (self.v[i*4+2]**0.5 + 1e-8)
            self.m[i*4+3] = 0.9 * self.m[i*4+3] + 0.1 * p.db
            self.v[i*4+3] = 0.999 * self.v[i*4+3] + 0.001 * p.db**2
            p.b -= r * self.m[i*4+3] / (self.v[i*4+3]**0.5 + 1e-8)

X, Y, u = np.random.randn(100, 784).astype('f4'), np.random.randint(10, size=100), 100
m = Model()
for e in range(101):
    z = m.forward(X)
    p = (v := np.exp(z - z.max(1, keepdims=1))) / v.sum(1, keepdims=1)
    ls = -np.log(p[range(100), Y] + 1e-9).mean()
    d = p.copy(); d[range(100), Y] -= 1
    m.backward(d, u)
    if e % 10 == 0: print(f"E:{e} L:{ls:.2f} A:{(z.argmax(1)==Y).mean():.2f}")