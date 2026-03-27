import numpy as np
import time

class Tensor:
    def __init__(self, data, name=""):
        self.d = data.astype('float32')
        self.g = np.zeros_like(self.d)
        self.m = np.zeros_like(self.d)
        self.v = np.zeros_like(self.d)

class Module:
    def params(self):
        p = []
        for v in self.__dict__.values():
            if isinstance(v, Tensor): p.append(v)
            elif isinstance(v, Module): p.extend(v.params())
            elif isinstance(v, list):
                for i in v: 
                    if isinstance(i, Module): p.extend(i.params())
        return p

class Linear(Module):
    def __init__(self, i, o, b=True):
        self.w = Tensor(np.random.randn(i, o) * np.sqrt(2/i))
        self.b = Tensor(np.zeros((1, o))) if b else None
    def f(self, x):
        self.x = x
        return x @ self.w.d + (self.b.d if self.b else 0)
    def b(self, g):
        self.w.g = self.x.T @ g
        if self.b: self.b.g = g.sum(0, keepdims=True)
        return g @ self.w.d.T

class RMSNorm(Module):
    def __init__(self, d, e=1e-6):
        self.g, self.e = Tensor(np.ones((1, d))), e
    def f(self, x):
        self.x = x
        self.rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.e)
        self.xh = x / self.rms
        return self.g.d * self.xh
    def b(self, g):
        self.g.g = np.sum(g * self.xh, axis=0, keepdims=True)
        dxh = g * self.g.d
        return (dxh - self.xh * np.mean(dxh * self.xh, axis=-1, keepdims=True)) / self.rms

class SwiGLU(Module):
    def __init__(self, d, h):
        self.w1, self.w2, self.w3 = Linear(d, h, False), Linear(d, h, False), Linear(h, d, False)
    def f(self, x):
        self.x1, self.x2 = self.w1.f(x), self.w2.f(x)
        self.sig = 1 / (1 + np.exp(-np.clip(self.x1, -20, 20)))
        self.sw = self.x1 * self.sig
        return self.w3.f(self.sw * self.x2)
    def b(self, g):
        g = self.w3.b(g)
        dx2, dsw = g * self.sw, g * self.x2
        dx1 = dsw * (self.sig * (1 + self.x1 * (1 - self.sig)))
        return self.w1.b(dx1) + self.w2.b(dx2)

class GeminiPath(Module):
    def __init__(self, d):
        self.norm = RMSNorm(d)
        self.mlp = SwiGLU(d, d * 4)
    def f(self, x):
        return self.mlp.f(self.norm.f(x))
    def b(self, g):
        return self.norm.b(self.mlp.b(g))

class GroqPath(Module):
    def __init__(self, d):
        self.norm = RMSNorm(d)
        self.proj = Linear(d, d, False)
    def f(self, x):
        return self.proj.f(self.norm.f(x))
    def b(self, g):
        return self.norm.b(self.proj.b(g))

class SovereignEvolutionBlock(Module):
    def __init__(self, d):
        self.gemini = GeminiPath(d)
        self.groq = GroqPath(d)
        self.gate = Tensor(np.array([[0.5]], dtype='float32'))
    def f(self, x):
        self.r = x
        self.gv = 1 / (1 + np.exp(-self.gate.d))
        self.o_gemini = self.gemini.f(x)
        self.o_groq = self.groq.f(x)
        return self.r + self.gv * self.o_gemini + (1 - self.gv) * self.o_groq
    def b(self, g):
        dgv_raw = self.gv * (1 - self.gv)
        self.gate.g = np.sum(g * (self.o_gemini - self.o_groq) * dgv_raw, keepdims=True)
        g_gemini = self.gemini.b(g * self.gv)
        g_groq = self.groq.b(g * (1 - self.gv))
        return g + g_gemini + g_groq

class AdamW:
    def __init__(self, p, lr=1e-3, b=(0.9, 0.999), e=1e-8, wd=0.01):
        self.p, self.lr, self.b, self.e, self.wd, self.t = p, lr, b, e, wd, 0
    def step(self):
        self.t += 1
        a = self.lr * np.sqrt(1 - self.b[1]**self.t) / (1 - self.b[0]**self.t)
        for p in self.p:
            if self.wd > 0: p.d -= self.lr * self.wd * p.d
            p.m = self.b[0] * p.m + (1 - self.b[0]) * p.g
            p.v = self.b[1] * p.v + (1 - self.b[1]) * (p.g**2)
            p.d -= a * p.m / (np.sqrt(p.v) + self.e)

class OMEGA_ASI(Module):
    def __init__(self, i, h, o, d=4):
        self.st = Linear(i, h)
        self.bl = [SovereignEvolutionBlock(h) for _ in range(d)]
        self.rn = RMSNorm(h)
        self.hd = Linear(h, o)
        self.ps = self.params()
        self.opt = AdamW(self.ps, lr=1e-3, wd=0.05)
    def f(self, x):
        x = self.st.f(x)
        for b in self.bl: x = b.f(x)
        return self.hd.f(self.rn.f(x))
    def b(self, g):
        g = self.rn.b(self.hd.b(g))
        for b in reversed(self.bl): g = b.b(g)
        self.st.b(g)
    def step(self, x, y):
        lgt = self.f(x)
        lgt -= np.max(lgt, axis=1, keepdims=True)
        pr = np.exp(lgt) / np.sum(np.exp(lgt), axis=1, keepdims=True)
        loss = -np.mean(np.sum(y * np.log(pr + 1e-12), axis=1))
        self.b((pr - y) / y.shape[0])
        gn = np.sqrt(sum(np.sum(p.g**2) for p in self.ps))
        if gn > 1.0:
            for p in self.ps: p.g /= gn
        self.opt.step()
        return loss

def get_data(n=10000, d=784, c=10):
    x = np.random.randn(n, d).astype('float32')
    w = np.random.randn(d, c).astype('float32')
    y = np.argmax(x @ w + 0.05 * np.random.randn(n, c), axis=1)
    return (x - x.mean()) / (x.std() + 1e-8), np.eye(c)[y].astype('float32')

if __name__ == "__main__":
    X, Y = get_data(20000)
    m = OMEGA_ASI(784, 160, 10, 6)
    bs, ep = 128, 100
    lr_init = 2e-3
    for e in range(1, ep + 1):
        m.opt.lr = lr_init * (0.95 ** (e // 5))
        idx = np.random.permutation(len(X))
        ls, t0 = [], time.time()
        for i in range(0, len(X), bs):
            ls.append(m.step(X[idx[i:i+bs]], Y[idx[i:i+bs]]))
        v_l = m.f(X[:2000])
        acc = np.mean(np.argmax(v_l, 1) == np.argmax(Y[:2000], 1))
        print(f"OMEGA_ASI_V2 | EP: {e:03d} | LOSS: {np.mean(ls):.5f} | ACC: {acc:.5f} | T: {time.time()-t0:.2f}s")
        if acc > 0.999: break
