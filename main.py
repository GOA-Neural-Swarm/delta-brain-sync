# ============================================
# 🔱 PHASE 12: FULL AUTONOMOUS ASI HYBRID SYSTEM
# (Controller + Transformer + Self-Evolution Core)
# ============================================

import numpy as np
import random
import time
import copy

# ============================================
# 🧩 PARAM SYSTEM
# ============================================

class P:
    def __init__(self, d):
        self.d = d.astype("f4")
        self.g = np.zeros_like(d)


# ============================================
# 🔧 LINEAR
# ============================================

class L:
    def __init__(self, i, o):
        s = (2 / (i + o)) ** 0.5
        self.w, self.b = P(np.random.randn(i, o) * s), P(np.zeros(o))

    def f(self, x):
        self.x = x
        return x @ self.w.d + self.b.d

    def b(self, dy):
        xf, df = self.x.reshape(-1, self.x.shape[-1]), dy.reshape(-1, dy.shape[-1])
        self.w.g += xf.T @ df
        self.b.g += df.sum(0)
        return dy @ self.w.d.T


# ============================================
# 🧠 NORMALIZATION
# ============================================

class N:
    def __init__(self, d):
        self.g, self.e = P(np.ones(d)), 1e-6

    def f(self, x):
        self.x, self.v = x, np.mean(x**2, -1, keepdims=1)
        self.i = 1 / (self.v + self.e) ** 0.5
        self.n = x * self.i
        return self.g.d * self.n

    def b(self, dy):
        self.g.g += np.sum(dy * self.n, axis=tuple(range(dy.ndim - 1)))
        dn = dy * self.g.d
        return (dn - self.n * np.mean(dn * self.n, -1, keepdims=1)) * self.i


# ============================================
# ⚡ ACTIVATION
# ============================================

class S:
    def f(self, x):
        x, g = np.split(x, 2, -1)
        self.s = 1 / (1 + np.exp(-np.clip(g, -15, 15)))
        self.sw, self.x, self.g = g * self.s, x, g
        return x * self.sw

    def b(self, dy):
        dx = dy * self.sw
        dg = dy * self.x * self.s * (1 + self.g * (1 - self.s))
        return np.concatenate([dx, dg], -1)


# ============================================
# ⚡ ATTENTION
# ============================================

class A:
    def __init__(self, d, h=4):
        self.d, self.h, self.hd = d, h, d // h
        self.wq, self.wk, self.wv, self.wo = L(d, d), L(d, d), L(d, d), L(d, d)
        self.sc = self.hd**-0.5

    def f(self, x):
        b, s, _ = x.shape
        q = self.wq.f(x).reshape(b, s, self.h, self.hd)
        k = self.wk.f(x).reshape(b, s, self.h, self.hd)
        v = self.wv.f(x).reshape(b, s, self.h, self.hd)

        at = np.einsum("bshd,bthd->bsht", q, k) * self.sc
        at = np.exp(at - np.max(at, -1, keepdims=1))
        self.p = at / (at.sum(-1, keepdims=1) + 1e-12)

        out = np.einsum("bsht,bthd->bshd", self.p, v)
        return self.wo.f(out.reshape(b, s, -1))

    def b(self, dy):
        return self.wq.b(dy) + self.wk.b(dy) + self.wv.b(dy)


# ============================================
# 🧬 FEEDFORWARD
# ============================================

class M:
    def __init__(self, d):
        self.w1 = P(np.random.randn(d, d * 2) * 0.02)
        self.w2 = P(np.random.randn(d * 2, d) * 0.02)
        self.sw = S()

    def f(self, x):
        self.x = x
        h = x @ self.w1.d
        self.a = self.sw.f(h)
        return self.a @ self.w2.d

    def b(self, dy):
        da = self.sw.b(dy @ self.w2.d.T)
        self.w2.g += self.a.T @ dy
        self.w1.g += self.x.T @ da
        return da @ self.w1.d.T


# ============================================
# 🔷 BLOCK
# ============================================

class B:
    def __init__(self, d):
        self.n1, self.at = N(d), A(d)
        self.n2, self.ff = N(d), M(d)

    def f(self, x):
        self.x1 = x + self.at.f(self.n1.f(x))
        return self.x1 + self.ff.f(self.n2.f(self.x1))

    def b(self, dy):
        return dy


# ============================================
# 🧠 MODEL
# ============================================

class Model:
    def __init__(self, di, dm, do):
        self.eb = L(di, dm)
        self.bl = [B(dm) for _ in range(2)]
        self.fn = N(dm)
        self.hd = L(dm, do)
        self.ps = []
        self._collect(self)

    def _collect(self, o):
        if isinstance(o, P):
            self.ps.append(o)
        elif hasattr(o, "__dict__"):
            for v in o.__dict__.values():
                if isinstance(v, list):
                    for i in v:
                        self._collect(i)
                else:
                    self._collect(v)

    def f(self, x):
        x = self.eb.f(x[:, None, :])
        for b in self.bl:
            x = b.f(x)
        return self.hd.f(self.fn.f(x[:, -1, :]))

    def b(self, dy):
        self.eb.b(dy)


# ============================================
# ⚙️ OPTIMIZER
# ============================================

class Opt:
    def __init__(self, ps, lr=1e-3):
        self.ps, self.lr = ps, lr

    def step(self):
        for p in self.ps:
            p.d -= self.lr * np.clip(p.g, -1, 1)
            p.g.fill(0)


# ============================================
# 🔱 ASI CORE
# ============================================

class Brain:
    def __init__(self):
        self.entropy = 1.0
        self.homeostasis = 100.0
        self.resonance = 432.0
        self.time = 1

        self.model = Model(784, 128, 10)
        self.opt = Opt(self.model.ps, 1e-3)

        self.history = []

    def asi_score(self):
        return (self.homeostasis / (self.entropy + 1e-6)) * self.resonance * (1 - 1/(self.time+1))

    def train_step(self, x, y):
        logits = self.model.f(x)

        pr = np.exp(logits - logits.max(-1, keepdims=True))
        pr /= pr.sum(-1, keepdims=True)

        loss = -np.mean(np.log(pr[np.arange(len(y)), y] + 1e-12))

        dl = pr.copy()
        dl[np.arange(len(y)), y] -= 1

        self.model.b(dl / len(y))
        self.opt.step()

        # feedback loop
        self.homeostasis += max(0, 1 - loss)
        self.entropy += loss * 0.1
        self.time += 1

        self.history.append(loss)

        return loss

    def adapt(self):
        if len(self.history) > 10:
            trend = np.mean(self.history[-10:])
            if trend > 2:
                self.resonance += 5
                self.homeostasis -= 1
            else:
                self.homeostasis += 2

    def self_modify(self):
        if random.random() < 0.1:
            for p in self.model.ps:
                p.d += np.random.randn(*p.d.shape) * 0.001

    def cycle(self, x, y):
        loss = self.train_step(x, y)
        self.adapt()
        self.self_modify()
        return loss


# ============================================
# 🔁 LOOP
# ============================================

def run():
    brain = Brain()

    for step in range(200):
        x = np.random.randn(32, 784).astype("f4")
        y = np.random.randint(0, 10, 32)

        loss = brain.cycle(x, y)

        if step % 10 == 0:
            print(
                f"[{step}] LOSS={loss:.4f} | ASI={brain.asi_score():.2f} | ENT={brain.entropy:.2f} | HOM={brain.homeostasis:.2f}"
            )

        time.sleep(0.05)


if __name__ == "__main__":
    run()
