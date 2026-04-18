import numpy as np

class Tensor:
    def __init__(self, data):
        self.data = data.astype("f4")
        self.grad = np.zeros_like(self.data)

class Module:
    def params(self):
        p = []
        for v in self.__dict__.values():
            if isinstance(v, Tensor): p.append(v)
            elif isinstance(v, Module): p.extend(v.params())
            elif isinstance(v, list): [p.extend(i.params()) for i in v if isinstance(i, Module)]
        return p

class Linear(Module):
    def __init__(self, i, o, bias=False):
        self.w = Tensor(np.random.randn(i, o) * np.sqrt(2./i))
        self.b = Tensor(np.zeros(o)) if bias else None

    def forward(self, x):
        self.x = x
        return x @ self.w.data + (self.b.data if self.b else 0)

    def backward(self, dy):
        x_f, dy_f = self.x.reshape(-1, self.x.shape[-1]), dy.reshape(-1, dy.shape[-1])
        self.w.grad += x_f.T @ dy_f
        if self.b: self.b.grad += dy_f.sum(0)
        return dy @ self.w.data.T

class RMSNorm(Module):
    def __init__(self, d, e=1e-6):
        self.g, self.e = Tensor(np.ones(d)), e

    def forward(self, x):
        self.x = x
        self.v = np.mean(x**2, -1, keepdims=True)
        self.inv = 1.0 / np.sqrt(self.v + self.e)
        self.nx = x * self.inv
        return self.g.data * self.nx

    def backward(self, dy):
        dg = dy * self.g.data
        self.g.grad += np.sum(dy * self.nx, axis=tuple(range(dy.ndim - 1)))
        return (dg - self.nx * np.mean(dg * self.nx, -1, keepdims=True)) * self.inv

class GQA(Module):
    def __init__(self, d, h=8, g=2):
        self.d, self.h, self.g, self.hd = d, h, g, d // h
        self.wq, self.wk = Linear(d, d), Linear(d, (h // g) * self.hd)
        self.wv, self.wo = Linear(d, (h // g) * self.hd), Linear(d, d)
        self.scale = self.hd**-0.5

    def _rope(self, t, inv=False):
        b, s, h, d = t.shape
        f = 10000**-(np.arange(0, d, 2)/d)
        a = np.arange(s)[:, None] * f
        cos, sin = np.cos(a), np.sin(a) * (-1 if inv else 1)
        r, i = t[..., ::2], t[..., 1::2]
        out = np.empty_like(t)
        out[..., ::2], out[..., 1::2] = r * cos[:, None, :] - i * sin[:, None, :], r * sin[:, None, :] + i * cos[:, None, :]
        return out

    def forward(self, x):
        b, s, _ = x.shape
        q = self.wq.forward(x).reshape(b, s, self.h, self.hd)
        k = self.wk.forward(x).reshape(b, s, self.h // self.g, self.hd)
        v = self.wv.forward(x).reshape(b, s, self.h // self.g, self.hd)
        self.qr, self.kr, self.vr = self._rope(q), self._rope(k), v
        kr_r, vr_r = np.repeat(self.kr, self.g, 2), np.repeat(self.vr, self.g, 2)
        att = np.einsum("bshd,bthd->bsht", self.qr, kr_r) * self.scale
        self.p = np.exp(att - np.max(att, -1, keepdims=True))
        self.p /= self.p.sum(-1, keepdims=True) + 1e-12
        return self.wo.forward(np.einsum("bsht,bthd->bshd", self.p, vr_r).reshape(b, s, -1))

    def backward(self, dy):
        b, s = dy.shape[:2]
        dy_wo = self.wo.backward(dy).reshape(b, s, self.h, self.hd)
        kr_r, vr_r = np.repeat(self.kr, self.g, 2), np.repeat(self.vr, self.g, 2)
        dp = np.einsum("bshd,bthd->bsht", dy_wo, vr_r)
        da = self.p * (dp - np.sum(self.p * dp, -1, keepdims=True)) * self.scale
        dqr = np.einsum("bsht,bthd->bshd", da, kr_r)
        dkr = np.einsum("bsht,bshd->bthd", da, self.qr).reshape(b, s, self.h//self.g, self.g, self.hd).sum(3)
        dvr = np.einsum("bsht,bshd->bthd", self.p, dy_wo).reshape(b, s, self.h//self.g, self.g, self.hd).sum(3)
        return self.wq.backward(self._rope(dqr, 1).reshape(b, s, -1)) + self.wk.backward(self._rope(dkr, 1).reshape(b, s, -1)) + self.wv.backward(dvr.reshape(b, s, -1))

class MoE(Module):
    def __init__(self, d, n=4, k=2):
        self.d, self.n, self.k = d, n, k
        self.gate = Linear(d, n)
        self.w1 = [Linear(d, d*2) for _ in range(n)]
        self.w2 = [Linear(d*2, d) for _ in range(n)]

    def forward(self, x):
        self.sh = x.shape
        xf = x.reshape(-1, self.d)
        lg = self.gate.forward(xf)
        p = np.exp(lg - np.max(lg, -1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        self.idx = np.argsort(p, -1)[:, -self.k:]
        self.w = np.take_along_axis(p, self.idx, -1)
        self.w /= self.w.sum(-1, keepdims=True) + 1e-12
        out, self.cache = np.zeros_like(xf), []
        for i in range(self.n):
            m = np.any(self.idx == i, -1)
            if not np.any(m): self.cache.append(None); continue
            pos = np.where(self.idx[m] == i)[1]
            h1 = self.w1[i].forward(xf[m])
            sig = 1 / (1 + np.exp(-np.clip(h1, -15, 15)))
            act = h1 * sig
            h2 = self.w2[i].forward(act)
            out[m] += h2 * self.w[m, pos][:, None]
            self.cache.append((m, pos, act, sig, h2))
        return out.reshape(self.sh)

    def backward(self, dy):
        dyf = dy.reshape(-1, self.d)
        dx, dg = np.zeros((dyf.shape[0], self.d)), np.zeros((dyf.shape[0], self.n))
        for i in range(self.n):
            if self.cache[i] is None: continue
            m, pos, act, sig, h2 = self.cache[i]
            dg[m, i] = np.sum(dyf[m] * h2, -1)
            ds = self.w2[i].backward(dyf[m] * self.w[m, pos][:, None])
            dx[m] += self.w1[i].backward(ds * sig * (1 + act * (1 - sig)))
        return (dx + self.gate.backward(dg - np.mean(dg, -1, keepdims=True))).reshape(self.sh)

class SovereignFusion(Module):
    def __init__(self, d):
        self.moe, self.gqa, self.gate = MoE(d), GQA(d, 4, 2), Linear(d, 2)

    def forward(self, x):
        self.om, self.og = self.moe.forward(x), self.gqa.forward(x)
        p = np.exp((lg := self.gate.forward(x)) - np.max(lg, -1, keepdims=True))
        self.p = p / (p.sum(-1, keepdims=True) + 1e-12)
        return self.p[..., :1] * self.om + self.p[..., 1:] * self.og

    def backward(self, dy):
        dp = np.stack([np.sum(dy * self.om, -1), np.sum(dy * self.og, -1)], -1)
        return self.moe.backward(dy * self.p[..., :1]) + self.gqa.backward(dy * self.p[..., 1:]) + self.gate.backward(dp - np.mean(dp, -1, keepdims=True))

class SovereignBlock(Module):
    def __init__(self, d):
        self.n1, self.at = RMSNorm(d), GQA(d)
        self.n2, self.ff = RMSNorm(d), SovereignFusion(d)

    def forward(self, x):
        x = x + self.at.forward(self.n1.forward(x))
        return x + self.ff.forward(self.n2.forward(x))

    def backward(self, dy):
        dy = dy + self.ff.backward(self.n2.backward(dy))
        return dy + self.at.backward(self.n1.backward(dy))

class OMEGA_ASI(Module):
    def __init__(self, di, dm, do, depth=2):
        self.embed = Linear(di, dm)
        self.blocks = [SovereignBlock(dm) for _ in range(depth)]
        self.norm, self.head = RMSNorm(dm), Linear(dm, do)

    def forward(self, x):
        x = self.embed.forward(x[:, None] if x.ndim == 2 else x)
        for b in self.blocks: x = b.forward(x)
        return self.head.forward(self.norm.forward(x[:, -1]))

    def backward(self, dy):
        dy = self.norm.backward(self.head.backward(dy))
        db = np.zeros((dy.shape[0], self.embed.x.shape[1], dy.shape[1]))
        db[:, -1] = dy
        for b in reversed(self.blocks): db = b.backward(db)
        self.embed.backward(db)

class AdamW:
    def __init__(self, p, lr=1e-3, wd=0.01, b1=0.9, b2=0.999):
        self.p, self.lr, self.wd, self.b1, self.b2 = p, lr, wd, b1, b2
        self.m, self.v, self.t = [np.zeros_like(i.data) for i in p], [np.zeros_like(i.data) for i in p], 0

    def step(self):
        self.t += 1
        lrt = self.lr * (np.sqrt(1 - self.b2**self.t) / (1 - self.b1**self.t))
        for i, pt in enumerate(self.p):
            g = np.clip(pt.grad, -1, 1)
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g**2)
            pt.data -= lrt * (self.m[i] / (np.sqrt(self.v[i]) + 1e-8) + self.wd * pt.data)
            pt.grad.fill(0)

def train():
    N, D, C, BS, E = 512, 784, 10, 32, 50
    X, Y = np.random.randn(N, D).astype("f4"), np.random.randint(0, C, N)
    m = OMEGA_ASI(D, 64, C, 2)
    opt = AdamW(m.params())
    for e in range(E):
        idx, ls, ac = np.random.permutation(N), [], []
        for i in range(0, N, BS):
            xb, yb = X[idx[i:i+BS]], Y[idx[i:i+BS]]
            lg = m.forward(xb)
            pr = (p := np.exp(lg - lg.max(-1, keepdims=True))) / p.sum(-1, keepdims=True)
            ls.append(-np.mean(np.log(pr[range(len(yb)), yb] + 1e-12)))
            ac.append(np.mean(pr.argmax(-1) == yb))
            dl = pr.copy(); dl[range(len(yb)), yb] -= 1
            m.backward(dl / len(yb)); opt.step()
        if (e + 1) % 10 == 0: print(f"E{e+1} | L:{np.mean(ls):.3f} | A:{np.mean(ac):.3f}")

if __name__ == "__main__": train()