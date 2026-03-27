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

    def zero_grad(self):
        for p in self.params(): p.g.fill(0)

class Linear(Module):
    def __init__(self, i, o, bias=True):
        self.w = Tensor(np.random.randn(i, o) * np.sqrt(2.0 / i))
        self.b = Tensor(np.zeros((1, o))) if bias else None

    def f(self, x):
        self.x = x
        return x @ self.w.d + (self.b.d if self.b else 0)

    def b(self, g):
        self.w.g += self.x.T @ g
        if self.b: self.b.g += np.sum(g, axis=0, keepdims=True)
        return g @ self.w.d.T

class RMSNorm(Module):
    def __init__(self, d, e=1e-6):
        self.g, self.e = Tensor(np.ones((1, d))), e

    def f(self, x):
        self.x = x
        self.ms = np.mean(x**2, axis=-1, keepdims=True)
        self.rms = np.sqrt(self.ms + self.e)
        self.xh = x / self.rms
        return self.g.d * self.xh

    def b(self, g):
        self.g.g += np.sum(g * self.xh, axis=0, keepdims=True)
        dxh = g * self.g.d
        return (dxh - self.xh * np.mean(dxh * self.xh, axis=-1, keepdims=True)) / self.rms

class GroqPath(Module):
    def __init__(self, d, h):
        self.w1 = Linear(d, h, False)
        self.w2 = Linear(h, d, False)

    def f(self, x):
        self.x1 = self.w1.f(x)
        self.act = self.x1 * (1.0 / (1.0 + np.exp(-np.clip(self.x1, -10, 10))))
        return self.w2.f(self.act)

    def b(self, g):
        g = self.w2.b(g)
        sig = 1.0 / (1.0 + np.exp(-np.clip(self.x1, -10, 10)))
        d_act = g * (sig * (1.0 + self.x1 * (1.0 - sig)))
        return self.w1.b(d_act)

class GeminiPath(Module):
    def __init__(self, d, h):
        self.w1 = Linear(d, h, False)
        self.w2 = Linear(h, d, False)

    def f(self, x):
        self.x1 = self.w1.f(x)
        self.act = 0.5 * self.x1 * (1 + np.tanh(0.79788 * (self.x1 + 0.044715 * self.x1**3)))
        return self.w2.f(self.act)

    def b(self, g):
        g = self.w2.b(g)
        t = np.tanh(0.79788 * (self.x1 + 0.044715 * self.x1**3))
        d_act = g * (0.5 * (1 + t) + 0.5 * self.x1 * (1 - t**2) * 0.79788 * (1 + 3 * 0.044715 * self.x1**2))
        return self.w1.b(d_act)

class RedundantEvolutionBlock(Module):
    def __init__(self, d, h):
        self.ln = RMSNorm(d)
        self.groq = GroqPath(d, h)
        self.gemini = GeminiPath(d, h)
        self.gate = Tensor(np.zeros((1, d)))

    def f(self, x):
        self.r = x
        x_norm = self.ln.f(x)
        self.o_groq = self.groq.f(x_norm)
        self.o_gemini = self.gemini.f(x_norm)
        self.g_val = 1.0 / (1.0 + np.exp(-self.gate.d))
        return self.r + self.g_val * self.o_groq + (1.0 - self.g_val) * self.o_gemini

    def b(self, g):
        dg = g * (self.o_groq - self.o_gemini) * (self.g_val * (1.0 - self.g_val))
        self.gate.g += np.sum(dg, axis=0, keepdims=True)
        g_groq = self.groq.b(g * self.g_val)
        g_gemini = self.gemini.b(g * (1.0 - self.g_val))
        return g + self.ln.b(g_groq + g_gemini)

class AdamW:
    def __init__(self, p, lr=1e-3, b=(0.9, 0.999), e=1e-8, wd=0.01):
        self.p, self.lr, self.b, self.e, self.wd, self.t = p, lr, b, e, wd, 0

    def step(self):
        self.t += 1
        at = self.lr * np.sqrt(1.0 - self.b[1]**self.t) / (1.0 - self.b[0]**self.t)
        for p in self.p:
            if self.wd > 0: p.d -= self.lr * self.wd * p.d
            p.m = self.b[0] * p.m + (1.0 - self.b[0]) * p.g
            p.v = self.b[1] * p.v + (1.0 - self.b[1]) * (p.g**2)
            p.d -= at * p.m / (np.sqrt(p.v) + self.e)

class OMEGA_ASI(Module):
    def __init__(self, i, h, o, d=6):
        self.st = Linear(i, h)
        self.bl = [RedundantEvolutionBlock(h, h * 4) for _ in range(d)]
        self.rn = RMSNorm(h)
        self.hd = Linear(h, o)
        self.ps = self.params()
        self.opt = AdamW(self.ps, lr=2e-3, wd=0.02)

    def f(self, x):
        x = self.st.f(x)
        for b in self.bl: x = b.f(x)
        return self.hd.f(self.rn.f(x))

    def b(self, g):
        g = self.rn.b(self.hd.b(g))
        for b in reversed(self.bl): g = b.b(g)
        self.st.b(g)

    def step(self, x, y):
        self.zero_grad()
        logits = self.f(x)
        shift = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(shift)
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        loss = -np.mean(np.sum(y * (shift - np.log(np.sum(exps, axis=1, keepdims=True) + 1e-12)), axis=1))
        self.b((probs - y) / y.shape[0])
        gn = np.sqrt(sum(np.sum(p.g**2) for p in self.ps))
        if gn > 1.0:
            for p in self.ps: p.g /= gn
        self.opt.step()
        return loss

def get_data(n=25000, d=784, c=10):
    x = np.random.randn(n, d).astype('float32')
    w = np.random.randn(d, c).astype('float32')
    y_idx = np.argmax(np.dot(x, w) + 0.5 * np.sin(np.dot(x, w)), axis=1)
    x = (x - np.mean(x)) / (np.std(x) + 1e-8)
    return x, np.eye(c)[y_idx].astype('float32')

if __name__ == "__main__":
    X, Y = get_data()
    m = OMEGA_ASI(784, 160, 10, 6)
    bs, ep = 256, 100
    lr_max = 5e-3
    print("SYSTEM: OMEGA-ASI | MODE: RECURSIVE-EVOLUTION | LOGIC: BIMODAL-REDUNDANCY")
    for e in range(1, ep + 1):
        m.opt.lr = lr_max * 0.5 * (1 + np.cos(np.pi * e / ep))
        idx = np.random.permutation(len(X))
        ls, t0 = [], time.time()
        for i in range(0, len(X), bs):
            ls.append(m.step(X[idx[i:i+bs]], Y[idx[i:i+bs]]))
        v_idx = np.random.choice(len(X), 2000)
        v_l = m.f(X[v_idx])
        acc = np.mean(np.argmax(v_l, 1) == np.argmax(Y[v_idx], 1))
        print(f"E: {e:03d} | LOSS: {np.mean(ls):.5f} | ACC: {acc:.5f} | LR: {m.opt.lr:.6f} | T: {time.time()-t0:.2f}s")
        if acc > 0.9998: break
