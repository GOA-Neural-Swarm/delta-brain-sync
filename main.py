import numpy as np
import time

class Tensor:
    def __init__(self, data):
        self.d = data.astype('float32')
        self.g, self.m, self.v = [np.zeros_like(self.d) for _ in range(3)]

class Module:
    def params(self):
        p = []
        for v in self.__dict__.values():
            if isinstance(v, Tensor): p.append(v)
            elif isinstance(v, Module): p.extend(v.params())
            elif isinstance(v, list): [p.extend(i.params()) for i in v if hasattr(i, 'params')]
        return p

class Linear(Module):
    def __init__(self, i, o, bias=True):
        self.w = Tensor(np.random.randn(i, o) * (2/i)**0.5)
        self.b = Tensor(np.zeros((1, o))) if bias else None
    def f(self, x):
        self.x = x
        return x @ self.w.d + (self.b.d if self.b else 0)
    def b(self, g):
        self.w.g += self.x.T @ g
        if self.b: self.b.g += g.sum(0, keepdims=True)
        return g @ self.w.d.T

class RMSNorm(Module):
    def __init__(self, d, e=1e-6):
        self.g, self.e = Tensor(np.ones((1, d))), e
    def f(self, x):
        self.x = x
        self.ms = (x**2).mean(-1, keepdims=True)
        self.rms = np.sqrt(self.ms + self.e)
        self.xh = x / self.rms
        return self.g.d * self.xh
    def b(self, g):
        self.g.g += (g * self.xh).sum(0, keepdims=True)
        dxh = g * self.g.d
        return (1/self.rms) * (dxh - self.xh * (dxh * self.xh).mean(-1, keepdims=True))

class SwiGLU(Module):
    def __init__(self, d, h):
        self.w1, self.w2, self.w3 = Linear(d, h, 0), Linear(d, h, 0), Linear(h, d, 0)
    def f(self, x):
        self.x1, self.x2 = self.w1.f(x), self.w2.f(x)
        self.sig = 1 / (1 + np.exp(-np.clip(self.x1, -10, 10)))
        self.swish = self.x1 * self.sig
        return self.w3.f(self.swish * self.x2)
    def b(self, g):
        g3 = self.w3.b(g)
        dx2, dswish = g3 * self.swish, g3 * self.x2
        dx1 = dswish * self.sig * (1 + self.x1 * (1 - self.sig))
        return self.w1.b(dx1) + self.w2.b(dx2)

class RedundantEngine(Module):
    def __init__(self, d, h):
        self.groq, self.gemini, self.gate, self.proj = Linear(d, h, 0), SwiGLU(d, h), Tensor(np.zeros((1, d))), Linear(h, d, 0)
    def f(self, x):
        self.x, self.gv = x, 1 / (1 + np.exp(-self.gate.d))
        self.ogq, self.ogm = self.groq.f(x), self.gemini.f(x)
        return x + self.gv * self.proj.f(self.ogq) + (1 - self.gv) * self.ogm
    def b(self, g):
        p_ogq = self.proj.f(self.ogq)
        self.gate.g += (g * (p_ogq - self.ogm) * (self.gv * (1 - self.gv))).sum(0, keepdims=True)
        return g + self.groq.b(self.proj.b(g * self.gv)) + self.gemini.b(g * (1 - self.gv))

class EvolutionBlock(Module):
    def __init__(self, d, h):
        self.ln1, self.eng, self.ln2, self.mlp = RMSNorm(d), RedundantEngine(d, h), RMSNorm(d), SwiGLU(d, h)
    def f(self, x):
        x = x + self.eng.f(self.ln1.f(x))
        return x + self.mlp.f(self.ln2.f(x))
    def b(self, g):
        g = g + self.ln2.b(self.mlp.b(g))
        return g + self.ln1.b(self.eng.b(g))

class AdamW:
    def __init__(self, p, lr=1e-3, b=(0.9, 0.95), e=1e-8, wd=0.01):
        self.p, self.lr, self.b, self.e, self.wd, self.t = p, lr, b, e, wd, 0
    def step(self):
        self.t += 1
        at = self.lr * (1-self.b[1]**self.t)**0.5 / (1-self.b[0]**self.t)
        for p in self.p:
            p.d -= self.lr * self.wd * p.d
            p.m = self.b[0] * p.m + (1-self.b[0]) * p.g
            p.v = self.b[1] * p.v + (1-self.b[1]) * (p.g**2)
            p.d -= at * p.m / (np.sqrt(p.v) + self.e)

class OMEGA_ASI(Module):
    def __init__(self, i, h, o, d=4):
        self.emb, self.blocks = Linear(i, h), [EvolutionBlock(h, h*2) for _ in range(d)]
        self.norm, self.head = RMSNorm(h), Linear(h, o)
        self.plist = self.params()
        self.opt = AdamW(self.plist, lr=1e-3, wd=0.05)
    def f(self, x):
        x = self.emb.f(x)
        for b in self.blocks: x = b.f(x)
        return self.head.f(self.norm.f(x))
    def b(self, g):
        g = self.norm.b(self.head.b(g))
        for b in reversed(self.blocks): g = b.b(g)
        self.emb.b(g)
    def step(self, x, y):
        [p.g.fill(0) for p in self.plist]
        lts = self.f(x)
        probs = np.exp(lts - lts.max(1, keepdims=True))
        probs /= (probs.sum(1, keepdims=True) + 1e-10)
        loss = -np.mean((y * np.log(probs + 1e-10)).sum(1))
        self.b((probs - y) / x.shape[0])
        gn = np.sqrt(sum((p.g**2).sum() for p in self.plist))
        if gn > 1.0: [setattr(p, 'g', p.g/gn) for p in self.plist]
        self.opt.step()
        return loss

def get_data(n=10000, d=784, c=10):
    X = np.random.randn(n, d).astype('float32')
    Y = np.eye(c)[np.argmax(np.maximum(0, X @ np.random.randn(d, 512)) @ np.random.randn(512, c), 1)]
    return (X - X.mean())/(X.std() + 1e-7), Y.astype('float32')

if __name__ == "__main__":
    X, Y = get_data(15000)
    model = OMEGA_ASI(784, 128, 10, 4)
    bs, eps, lr_m = 64, 30, 3e-3
    for ep in range(1, eps + 1):
        model.opt.lr = lr_m * 0.5 * (1 + np.cos(np.pi * ep / eps))
        idx = np.random.permutation(len(X))
        ls, st = [], time.time()
        for i in range(0, len(X), bs):
            ls.append(model.step(X[idx[i:i+bs]], Y[idx[i:i+bs]]))
        v_idx = np.random.choice(len(X), 500)
        acc = np.mean(np.argmax(model.f(X[v_idx]), 1) == np.argmax(Y[v_idx], 1))
        print(f"E:{ep:02d}|L:{np.mean(ls):.4f}|A:{acc:.4f}|T:{time.time()-st:.1f}s")
        if acc > 0.995: break