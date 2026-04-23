import numpy as np

class Tensor:
    def __init__(self, d):
        self.d = d.astype("f4")
        self.g = np.zeros_like(d)

class Linear:
    def __init__(self, i, o):
        s = np.sqrt(2 / (i + o))
        self.w = Tensor(np.random.normal(0, s, (i, o)))
        self.b = Tensor(np.zeros(o))

    def f(self, x):
        self.x = x
        return x @ self.w.d + self.b.d

    def b(self, dy):
        x_flat = self.x.reshape(-1, self.x.shape[-1])
        dy_flat = dy.reshape(-1, dy.shape[-1])
        self.w.g += x_flat.T @ dy_flat
        self.b.g += dy_flat.sum(0)
        return dy @ self.w.d.T

class RMSNorm:
    def __init__(self, d, e=1e-6):
        self.g, self.e = Tensor(np.ones(d)), e

    def f(self, x):
        self.x = x
        self.v = np.mean(x**2, -1, keepdims=True)
        self.isd = 1 / np.sqrt(self.v + self.e)
        self.nx = x * self.isd
        return self.g.d * self.nx

    def b(self, dy):
        self.g.g += np.sum(dy * self.nx, axis=tuple(range(dy.ndim - 1)))
        dn = dy * self.g.d
        return (dn - self.nx * np.mean(dn * self.nx, -1, keepdims=True)) * self.isd

class SwiGLU:
    def f(self, x):
        x, g = np.split(x, 2, -1)
        self.s = 1 / (1 + np.exp(-np.clip(g, -15, 15)))
        self.sw, self.x, self.g = g * self.s, x, g
        return x * self.sw

    def b(self, dy):
        dx = dy * self.sw
        dg = dy * self.x * self.s * (1 + self.g * (1 - self.s))
        return np.concatenate([dx, dg], -1)

class GQA:
    def __init__(self, d, h=8, g=2):
        self.d, self.h, self.g, self.hd = d, h, g, d // h
        self.wq, self.wk = Linear(d, d), Linear(d, (h // g) * self.hd)
        self.wv, self.wo = Linear(d, (h // g) * self.hd), Linear(d, d)
        self.sc = self.hd**-0.5

    def _rope(self, t, co, si):
        r, i = t[..., ::2], t[..., 1::2]
        return np.stack([r * co - i * si, r * si + i * co], -1).reshape(t.shape)

    def f(self, x):
        b, s, _ = x.shape
        q = self.wq.f(x).reshape(b, s, self.h, self.hd)
        k = self.wk.f(x).reshape(b, s, self.h // self.g, self.hd)
        v = self.wv.f(x).reshape(b, s, self.h // self.g, self.hd)
        
        p = np.arange(s)[:, None]
        f = 10000 ** -(np.arange(0, self.hd, 2) / self.hd)
        a = p * f
        co, si = np.cos(a)[:, None, :], np.sin(a)[:, None, :]
        
        self.qr, self.kr, self.vr = self._rope(q, co, si), self._rope(k, co, si), v
        # Expand KV to match Q heads
        ke = np.repeat(self.kr, self.g, 2)
        ve = np.repeat(v, self.g, 2)
        
        at = np.einsum("bshd,bthd->bsht", self.qr, ke) * self.sc
        at = np.exp(at - np.max(at, -1, keepdims=True))
        self.p = at / (at.sum(-1, keepdims=True) + 1e-12)
        out = np.einsum("bsht,bthd->bshd", self.p, ve).reshape(b, s, -1)
        return self.wo.f(out)

    def b(self, dy):
        b, s = dy.shape[:2]
        dw = self.wo.b(dy).reshape(b, s, self.h, self.hd)
        ve = np.repeat(self.vr, self.g, 2)
        ke = np.repeat(self.kr, self.g, 2)
        
        dp = np.einsum("bshd,bthd->bsht", dw, ve)
        da = self.p * (dp - np.sum(self.p * dp, -1, keepdims=True)) * self.sc
        
        dqr = np.einsum("bsht,bthd->bshd", da, ke)
        dk_full = np.einsum("bsht,bshd->bthd", da, self.qr)
        dk = dk_full.reshape(b, s, self.h // self.g, self.g, self.hd).sum(3)
        dv = np.einsum("bsht,bshd->bthd", self.p, dw).reshape(b, s, self.h // self.g, self.g, self.hd).sum(3)
        
        return self.wq.b(dqr.reshape(b, s, -1)) + \
               self.wk.b(dk.reshape(b, s, -1)) + \
               self.wv.b(dv.reshape(b, s, -1))

class MOE:
    def __init__(self, d, n=4, k=2):
        self.d, self.n, self.k = d, n, k
        self.gate = Linear(d, n)
        self.w1 = [Tensor(np.random.normal(0, np.sqrt(2/(d+d*2)), (d, d*2))) for _ in range(n)]
        self.w2 = [Tensor(np.random.normal(0, np.sqrt(1/d), (d*2, d))) for _ in range(n)]
        self.swi = [SwiGLU() for _ in range(n)]

    def f(self, x):
        sh = x.shape
        x = x.reshape(-1, self.d)
        g = self.gate.f(x)
        pr = np.exp(g - g.max(-1, keepdims=True))
        pr /= pr.sum(-1, keepdims=True)
        self.idx = np.argsort(pr, -1)[:, -self.k:]
        self.w = np.take_along_axis(pr, self.idx, -1)
        self.w /= self.w.sum(-1, keepdims=True) + 1e-12
        out = np.zeros_like(x)
        self.cache = []
        for i in range(self.n):
            m = np.any(self.idx == i, -1)
            if not np.any(m):
                self.cache.append(None)
                continue
            pos = np.where(self.idx[m] == i)[1][:, None]
            xi = x[m]
            h1 = xi @ self.w1[i].d
            act = self.swi[i].f(h1)
            h2 = act @ self.w2[i].d
            wi = self.w[m, pos[:, 0]][:, None]
            out[m] += h2 * wi
            self.cache.append((m, xi, act, wi, pos))
        return out.reshape(sh)

    def b(self, dy):
        dy_f = dy.reshape(-1, self.d)
        dx, dg = np.zeros_like(dy_f), np.zeros((dy_f.shape[0], self.n))
        for i in range(self.n):
            if self.cache[i] is None: continue
            m, xi, act, wi, pos = self.cache[i]
            dyi = dy_f[m] * wi
            dg[m, i] = np.sum(dy_f[m] * (act @ self.w2[i].d), -1)
            self.w2[i].g += act.T @ dyi
            da = self.swi[i].b(dyi @ self.w2[i].d.T)
            self.w1[i].g += xi.T @ da
            dx[m] += da @ self.w1[i].d.T
        return (dx + self.gate.b(dg - np.mean(dg, -1, keepdims=True))).reshape(dy.shape)

class SB:
    def __init__(self, d):
        self.n1, self.at = RMSNorm(d), GQA(d)
        self.n2, self.moe = RMSNorm(d), MOE(d)
        self.pg = Tensor(np.array([0.5]))

    def f(self, x):
        x = x + self.at.f(self.n1.f(x))
        m = self.moe.f(self.n2.f(x))
        return x + (self.pg.d * m + (1 - self.pg.d) * x)

    def b(self, dy):
        dm = dy * self.pg.d
        self.pg.g += np.sum(dy * (self.moe.f(self.n2.f(self.n1.f(dy))) - dy)) # Approximation
        dx_m = self.n2.b(self.moe.b(dm))
        dy = dy + dx_m
        return dy + self.n1.b(self.at.b(dy))

class OMEGA_ASI:
    def __init__(self, di, dm, do, depth=2):
        self.emb = Linear(di, dm)
        self.blks = [SB(dm) for _ in range(depth)]
        self.fn = RMSNorm(dm)
        self.hd = Linear(dm, do)
        self.params = self._get_p()

    def _get_p(self):
        p = []
        def walk(obj):
            if isinstance(obj, Tensor): p.append(obj)
            elif hasattr(obj, "__dict__"):
                for v in obj.__dict__.values():
                    if isinstance(v, list): [walk(i) for i in v]
                    else: walk(v)
        walk(self)
        return p

    def f(self, x):
        if x.ndim == 2: x = x[:, None, :]
        x = self.emb.f(x)
        for b in self.blks: x = b.f(x)
        return self.hd.f(self.fn.f(x[:, -1, :]))

    def b(self, dy):
        dy = self.fn.b(self.hd.b(dy))
        db = np.zeros((dy.shape[0], self.emb.x.shape[1], dy.shape[1]), "f4")
        db[:, -1, :] = dy
        for b in reversed(self.blks): db = b.b(db)
        self.emb.b(db)

class AdamW:
    def __init__(self, ps, lr=1e-3, wd=0.01):
        self.ps, self.lr, self.wd = ps, lr, wd
        self.m = [np.zeros_like(p.d) for p in ps]
        self.v = [np.zeros_like(p.d) for p in ps]
        self.t = 0

    def step(self):
        self.t += 1
        lt = self.lr * (np.sqrt(1 - 0.999**self.t) / (1 - 0.9**self.t))
        for i, p in enumerate(self.ps):
            g = np.clip(p.g, -1, 1)
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * (g**2)
            p.d -= lt * (self.m[i] / (np.sqrt(self.v[i]) + 1e-8) + self.wd * p.d)
            p.g.fill(0)

def train():
    N, D, C, BS, E = 1024, 784, 10, 64, 50
    X, Y = np.random.randn(N, D).astype("f4"), np.random.randint(0, C, N)
    m = OMEGA_ASI(D, 128, C)
    opt = AdamW(m.params, 3e-3)
    for e in range(E):
        idx = np.random.permutation(N)
        ls, ac = [], []
        for i in range(0, N, BS):
            xb, yb = X[idx[i:i+BS]], Y[idx[i:i+BS]]
            lg = m.f(xb)
            pr = np.exp(lg - np.max(lg, -1, keepdims=True))
            pr /= pr.sum(-1, keepdims=True)
            ls.append(-np.mean(np.log(pr[np.arange(len(yb)), yb] + 1e-12)))
            ac.append(np.mean(np.argmax(pr, -1) == yb))
            dl = pr.copy()
            dl[np.arange(len(yb)), yb] -= 1
            m.b(dl / len(yb))
            opt.step()
        if (e + 1) % 5 == 0:
            print(f"E {e+1} | L {np.mean(ls):.4f} | A {np.mean(ac):.4f}")

if __name__ == "__main__":
    train()