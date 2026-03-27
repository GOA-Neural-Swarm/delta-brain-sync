
import numpy as np
import time

class Tensor:
    def __init__(self, data):
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
        n = self.x.shape[-1]
        return (1.0 / self.rms) * (dxh - self.xh * np.mean(dxh * self.xh, axis=-1, keepdims=True))

class RedundantPath(Module):
    def __init__(self, d, h):
        self.w1 = Linear(d, h, False)
        self.w2 = Linear(h, d, False)
        self.gate = Tensor(np.zeros((1, d)))

    def f(self, x):
        self.z = self.w1.f(x)
        self.tanh_z = np.tanh(0.79788 * (self.z + 0.044715 * self.z**3))
        self.sig = 1.0 / (1.0 + np.exp(-np.clip(self.z, -10, 10)))
        self.act = self.z * self.sig
        self.act_g = 0.5 * self.z * (1.0 + self.tanh_z)
        self.gv = 1.0 / (1.0 + np.exp(-self.gate.d))
        self.o1 = self.w2.f(self.act)
        self.o2 = self.w2.f(self.act_g)
        return x + self.gv * self.o1 + (1.0 - self.gv) * self.o2

    def b(self, g):
        dg = g * (self.o1 - self.o2) * (self.gv * (1.0 - self.gv))
        self.gate.g += np.sum(dg, axis=0, keepdims=True)
        g1 = self.w2.b(g * self.gv)
        g2 = self.w2.b(g * (1.0 - self.gv))
        dsilu = self.sig * (1.0 + self.z * (1.0 - self.sig))
        dz = 0.5 * (1.0 + self.tanh_z) + 0.5 * self.z * (1.0 - self.tanh_z**2) * 0.79788 * (1.0 + 3.0 * 0.044715 * self.z**2)
        return g + self.w1.b(g1 * dsilu + g2 * dz)

class EvolutionBlock(Module):
    def __init__(self, d, h):
        self.ln = RMSNorm(d)
        self.path = RedundantPath(d, h)

    def f(self, x):
        self.r = x
        xn = self.ln.f(x)
        return self.path.f(xn)

    def b(self, g):
        g = self.path.b(g)
        return g + self.ln.b(g)

class AdamW:
    def __init__(self, p, lr=1e-3, b=(0.9, 0.999), e=1e-8, wd=0.01):
        self.p, self.lr, self.b, self.e, self.wd, self.t = p, lr, b, e, wd, 0

    def step(self):
        self.t += 1
        at = self.lr * np.sqrt(1.0 - self.b[1]**self.t) / (1.0 - self.b[0]**self.t)
        for p in self.p:
            p.d -= self.lr * self.wd * p.d
            p.m = self.b[0] * p.m + (1.0 - self.b[0]) * p.g
            p.v = self.b[1] * p.v + (1.0 - self.b[1]) * (p.g**2)
            p.d -= at * p.m / (np.sqrt(p.v) + self.e)

class OMEGA_ASI(Module):
    def __init__(self, i, h, o, d=6):
        self.st = Linear(i, h)
        self.bl = [EvolutionBlock(h, h * 4) for _ in range(d)]
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
        for p in self.ps: p.g.fill(0)
        logits = self.f(x)
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exps / (np.sum(exps, axis=1, keepdims=True) + 1e-12)
        loss = -np.mean(np.sum(y * np.log(probs + 1e-12), axis=1))
        self.b((probs - y) / y.shape[0])
        gn = np.sqrt(sum(np.sum(p.g**2) for p in self.ps))
        if gn > 1.0:
            for p in self.ps: p.g /= gn
        self.opt.step()
        return loss

def get_data(n=10000, d=784, c=10):
    x = np.random.randn(n, d).astype('float32')
    w = np.random.randn(d, c).astype('float32')
    y_idx = np.argmax(x @ w + 0.1 * np.sin(x @ w * 5), axis=1)
    x = (x - np.mean(x)) / (np.std(x) + 1e-8)
    return x, np.eye(c)[y_idx].astype('float32')

if __name__ == "__main__":
    X, Y = get_data(20000)
    m = OMEGA_ASI(784, 128, 10, 4)
    bs, ep = 128, 50
    lr_max = 4e-3
    print("SYSTEM: OMEGA-ASI | ARCHITECTURE: MODULAR-REDUNDANT | STATUS: EVOLVING")
    for e in range(1, ep + 1):
        m.opt.lr = lr_max * 0.5 * (1 + np.cos(np.pi * e / ep))
        idx = np.random.permutation(len(X))
        ls, t0 = [], time.time()
        for i in range(0, len(X), bs):
            ls.append(m.step(X[idx[i:i+bs]], Y[idx[i:i+bs]]))
        v_idx = np.random.choice(len(X), 1000)
        v_l = m.f(X[v_idx])
        acc = np.mean(np.argmax(v_l, 1) == np.argmax(Y[v_idx], 1))
        print(f"EPOCH: {e:02d} | LOSS: {np.mean(ls):.4f} | ACC: {acc:.4f} | LR: {m.opt.lr:.5f} | TIME: {time.time()-t0:.2f}s")
        if acc > 0.999: break
