
import numpy as np

class Tensor:
    def __init__(self, data):
        self.data = data.astype("f4")
        self.grad = np.zeros_like(self.data)

class Module:
    def params(self):
        p = []
        for v in self.__dict__.values():
            if isinstance(v, Tensor):
                p.append(v)
            elif isinstance(v, Module):
                p.extend(v.params())
            elif isinstance(v, list):
                for i in v:
                    p.extend(i.params() if isinstance(i, Module) else [i] if isinstance(i, Tensor) else [])
        return p

class Linear(Module):
    def __init__(self, i, o, b=True):
        self.w = Tensor(np.random.randn(i, o) * (2 / i) ** 0.5)
        self.b = Tensor(np.zeros(o)) if b else None

    def f(self, x):
        self.x = x
        return x @ self.w.data + (self.b.data if self.b else 0)

    def b(self, dy):
        xf = self.x.reshape(-1, self.x.shape[-1])
        df = dy.reshape(-1, dy.shape[-1])
        self.w.grad += xf.T @ df
        if self.b:
            self.b.grad += df.sum(0)
        return (dy @ self.w.data.T).reshape(self.x.shape)

class ReLU(Module):
    def f(self, x):
        self.x = x
        return np.maximum(x, 0)

    def b(self, dy):
        return dy * (self.x > 0)

class LayerNorm(Module):
    def __init__(self, d, e=1e-6):
        self.g, self.e = Tensor(np.ones(d)), e

    def f(self, x):
        self.x, self.v = x, np.mean(x**2, -1, keepdims=True)
        self.i = (self.v + self.e) ** -0.5
        self.nx = x * self.i
        return self.g.data * self.nx

    def b(self, dy):
        dg = dy * self.g.data
        self.g.grad += np.sum(dy * self.nx, axis=tuple(range(dy.ndim - 1)))
        return (dg - self.nx * np.mean(dg * self.nx, -1, keepdims=True)) * self.i

class Swish(Module):
    def f(self, x):
        self.x = x
        self.sig = 1 / (1 + np.exp(-np.clip(x, -12, 12)))
        return x * self.sig

    def b(self, dy):
        return dy * (self.sig * (1 + self.x * (1 - self.sig)))

class GELU(Module):
    def f(self, x):
        self.x = x
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    def b(self, dy):
        return dy * (0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (self.x + 0.044715 * np.power(self.x, 3)))))

class MultiHeadAttention(Module):
    def __init__(self, d, h=8, g=2):
        self.d, self.h, self.g, self.hd = d, h, g, d // h
        self.wq, self.wk, self.wv, self.wo = Linear(d, d), Linear(d, (h // g) * self.hd), Linear(d, (h // g) * self.hd), Linear(d, d)
        self.sc = self.hd**-0.5
        f = 10000 ** -(np.arange(0, self.hd, 2) / self.hd)
        t = np.arange(2048)[:, None] * f
        self.cos, self.sin = np.cos(t), np.sin(t)

    def _rp(self, x, v=False):
        s = x.shape[1]
        c, sn = self.cos[:s, None, :], self.sin[:s, None, :] * (-1 if v else 1)
        r, i = x[..., ::2], x[..., 1::2]
        o = np.empty_like(x)
        o[..., ::2], o[..., 1::2] = r * c - i * sn, r * sn + i * c
        return o

    def f(self, x):
        b, s, _ = x.shape
        q, k, v = self.wq.f(x).reshape(b, s, self.h, self.hd), self.wk.f(x).reshape(b, s, self.h // self.g, self.hd), self.wv.f(x).reshape(b, s, self.h // self.g, self.hd)
        self.qr, self.kr, self.v_o = self._rp(q), self._rp(k), v
        at = np.einsum("bshd,bthd->bsht", self.qr, np.repeat(self.kr, self.g, 2)) * self.sc
        self.p = (e := np.exp(at - at.max(-1, keepdims=True))) / (e.sum(-1, keepdims=True) + 1e-12)
        return self.wo.f(np.einsum("bsht,bthd->bshd", self.p, np.repeat(v, self.g, 2)).reshape(b, s, -1))

    def b(self, dy):
        b, s = dy.shape[:2]
        do = self.wo.b(dy).reshape(b, s, self.h, self.hd)
        dp = np.einsum("bshd,bthd->bsht", do, self.kv)
        da = self.p * (dp - (self.p * dp).sum(-1, keepdims=True)) * self.sc
        dq = self._rp(np.einsum("bsht,bthd->bshd", da, np.repeat(self.kr, self.g, 2)), True)
        dk = self._rp(np.einsum("bsht,bshd->bthd", da, self.qr).reshape(b, s, -1, self.g, self.hd).sum(3), True)
        dv = np.einsum("bsht,bshd->bthd", self.p, dy).reshape(b, s, -1, self.g, self.hd).sum(3)
        return self.wq.b(dq.reshape(b, s, -1)) + self.wk.b(dk.reshape(b, s, -1)) + self.wv.b(dv.reshape(b, s, -1))

class MixtureOfExperts(Module):
    def __init__(self, d, n=4, k=2):
        self.d, self.n, self.k, self.gate = d, n, k, Linear(d, n)

    def f(self, x):
        self.sh, xf = x.shape, x.reshape(-1, self.d)
        p = (e := np.exp(l := self.gate.f(xf) - self.gate.f(xf).max(-1, keepdims=True))) / (e.sum(-1, keepdims=True) + 1e-12)
        self.idx = np.argsort(p, -1)[:, -self.k:]
        self.w = np.take_along_axis(p, self.idx, -1)
        self.w /= self.w.sum(-1, keepdims=True) + 1e-12
        out, self.cache = np.zeros_like(xf), []
        for i in range(self.n):
            m = np.any(self.idx == i, -1)
            if not np.any(m):
                self.cache.append(None)
                continue
            ps = np.where(self.idx[m] == i)[1]
            h = self.experts[i][2].f(self.experts[i][1].f(self.experts[i][0].f(xf[m])))
            out[m] += h * self.w[m, ps][:, None]
            self.cache.append((m, ps, h))
        return out.reshape(self.sh)

    def b(self, dy):
        dyf, dx, dg = dy.reshape(-1, self.d), np.zeros((dy.size // self.d, self.d)), np.zeros((dy.size // self.d, self.n))
        for i in range(self.n):
            if self.cache[i] is not None:
                m, ps, h = self.cache[i]
                dg[m, i] = (dyf[m] * h).sum(-1)
                dx[m] += self.experts[i][0].b(self.experts[i][1].b(self.experts[i][2].b(dyf[m] * self.w[m, ps][:, None])))
        return (dx + self.gate.b(dg - dg.mean(-1, keepdims=True))).reshape(self.sh)

class DeepSlimLayer(Module):
    def __init__(self, d):
        self.p1, self.p2, self.fuz = Linear(d, d * 2), Swish(), Linear(d, d), Linear(d * 2, d)

    def f(self, x):
        self.o1 = self.p1[2].f(self.p1[1].f(self.p1[0].f(x)))
        self.o2 = self.p2[2].f(self.p2[1].f(self.p2[0].f(x)))
        return self.fuz.f(np.concatenate([self.o1, self.o2], -1))

    def b(self, dy):
        d1, d2 = np.split(self.fuz.b(dy), 2, -1)
        return self.p1[0].b(self.p1[1].b(self.p1[2].b(d1))) + self.p2[0].b(self.p2[1].b(self.p2[2].b(d2)))

class Block(Module):
    def __init__(self, d):
        self.n1, self.at, self.n2, self.ds, self.n3, self.mo = LayerNorm(d), MultiHeadAttention(d), LayerNorm(d), DeepSlimLayer(d), LayerNorm(d), MixtureOfExperts(d)

    def f(self, x):
        x = x + self.at.f(self.n1.f(x))
        x = x + self.ds.f(self.n2.f(x))
        return x + self.mo.f(self.n3.f(x))

    def b(self, dy):
        dy = dy + self.mo.b(self.n3.b(dy))
        dy = dy + self.ds.b(self.n2.b(dy))
        return dy + self.at.b(self.n1.b(dy))

class ASI(Module):
    def __init__(self, di, dm, do, dp=3):
        self.em, self.bl = Linear(di, dm), [Block(dm) for _ in range(dp)]
        self.nm, self.hd = LayerNorm(dm), Linear(dm, do)

    def f(self, x):
        x = self.em.f(x[:, None] if x.ndim == 2 else x)
        for b in self.bl:
            x = b.f(x)
        return self.hd.f(self.nm.f(x[:, -1]))

    def b(self, dy):
        dy = self.nm.b(self.hd.b(dy))
        db = np.zeros((dy.shape[0], self.em.x.shape[1], dy.shape[1]))
        db[:, -1] = dy
        for b in reversed(self.bl):
            db = b.b(db)
        return self.em.b(db)

class Adam:
    def __init__(self, p, lr=1e-3, b1=0.9, b2=0.999, wd=0.01):
        self.p, self.lr, self.b1, self.b2, self.wd, self.t = p, lr, b1, b2, wd, 0
        self.m, self.v = [np.zeros_like(i.data) for i in p], [np.zeros_like(i.data) for i in p]

    def step(self):
        self.t += 1
        a = self.lr * (np.sqrt(1 - self.b2**self.t) / (1 - self.b1**self.t))
        for i, pt in enumerate(self.p):
            g = np.clip(pt.grad, -10, 10)
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g**2)
            pt.data -= a * (self.m[i] / (np.sqrt(self.v[i]) + 1e-8) + self.wd * pt.data)
            pt.grad.fill(0)

if __name__ == "__main__":
    N, D, C, BS, E = 1024, 784, 10, 64, 50
    X, Y = np.random.randn(N, D), np.random.randint(0, C, N)
    m = ASI(D, 128, C, 2)
    opt = Adam(m.params(), 2e-3)
    for e in range(E):
        idx = np.random.permutation(N)
        losses, accs = [], []
        for i in range(0, N, BS):
            xb, yb = X[idx[i : i + BS]], Y[idx[i : i + BS]]
            lg = m.f(xb)
            pr = (ex := np.exp(lg - lg.max(-1, keepdims=True))) / (ex.sum(-1, keepdims=True) + 1e-12)
            ls = -np.mean(np.log(pr[range(len(yb)), yb] + 1e-12))
            ac = np.mean(pr.argmax(-1) == yb)
            losses.append(ls)
            accs.append(ac)
            dl = pr.copy()
            dl[range(len(yb)), yb] -= 1
            m.b(dl / len(yb))
            opt.step()
        if (e + 1) % 5 == 0:
            print(f"E {e+1:02} | L: {np.mean(losses):.4f} | A: {np.mean(accs):.4f}")
