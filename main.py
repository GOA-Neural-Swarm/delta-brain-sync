import numpy as np

class Normalization:
    def __init__(self, d):
        self.w, self.b = np.ones((1, d)), np.zeros((1, d))

    def forward(self, x):
        self.h = (x - x.mean(-1, keepdims=True)) / (x.var(-1, keepdims=True) + 1e-5)**0.5
        return self.w * self.h + self.b

    def backward(self, d):
        x = d * self.w
        self.dw, self.db = (d * self.h).sum(0, keepdims=True), d.sum(0, keepdims=True)
        return (x - x.mean(-1, keepdims=True) - self.h * (x * self.h).mean(-1, keepdims=True)) / (x.var(-1, keepdims=True) + 1e-5)**0.5

class Activation:
    def forward(self, x):
        self.s = 1 / (1 + np.exp(-np.clip(x, -20, 20)))
        self.x = x
        return x * self.s

    def backward(self, d):
        return d * (self.s + self.x * self.s * (1 - self.s))

class Linear:
    def __init__(self, i, o):
        self.w, self.b = np.random.randn(i, o) * (2 / i)**0.5, np.zeros((1, o))

    def forward(self, x):
        self.x = x
        return x @ self.w + self.b

    def backward(self, d):
        self.dw, self.db = self.x.T @ d, d.sum(0, keepdims=True)
        return d @ self.w.T

class Bottleneck:
    def __init__(self, d):
        self.layers = [Normalization(d), Linear(d, d), Activation(), Linear(d, d)]

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer.forward(h)
        return h + x

    def backward(self, d):
        g = d
        for layer in self.layers[::-1]:
            g = layer.backward(g)
        return d + g

class Model:
    def __init__(self, i=784, h=128, o=10, c=3):
        self.layers = [Linear(i, h), *([Bottleneck(h) for _ in range(c)]), Linear(h, o)]
        self.params = []
        for layer in self.layers:
            for sublayer in getattr(layer, 'layers', [layer]):
                if hasattr(sublayer, 'w'):
                    self.params.append(sublayer)
        self.moments = [0 * param.w for param in self.params] + [0 * param.b for param in self.params]
        self.variances = [0 * moment for moment in self.moments]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, d, u):
        for layer in self.layers[::-1]:
            d = layer.backward(d)
        self.t += 1
        r = 2e-3 * (1 - 0.999**self.t)**0.5 / (1 - 0.9**self.t)
        gradients = [param.dw / u for param in self.params] + [param.db / u for param in self.params]
        params = [param.w for param in self.params] + [param.b for param in self.params]
        for i in range(len(params)):
            self.moments[i] = 0.9 * self.moments[i] + 0.1 * gradients[i]
            self.variances[i] = 0.999 * self.variances[i] + 0.001 * gradients[i]**2
            params[i] -= r * self.moments[i] / (self.variances[i]**0.5 + 1e-8)

X, Y = np.random.randn(100, 784).astype('f4'), np.random.randint(10, size=100)
model = Model()
u = 100
t = 0
for e in range(101):
    z = model.forward(X)
    v = np.exp(z - z.max(1, keepdims=True))
    p = v / v.sum(1, keepdims=True)
    ls = -np.log(p[np.arange(100), Y] + 1e-9).mean()
    d = p.copy()
    d[np.arange(100), Y] -= 1
    model.backward(d, u)
    t += 1
    if e % 10 == 0:
        print(f"E:{e} L:{ls:.2f} A:{(z.argmax(1) == Y).mean():.2f}")