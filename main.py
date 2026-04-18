
import numpy as np

class Tensor:
    def __init__(self, data, name=""):
        self.data = data.astype("f4")
        self.grad = np.zeros_like(self.data)
        self.name = name

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
                    if isinstance(i, Module):
                        p.extend(i.params())
        return p

class Linear(Module):
    def __init__(self, i, o, bias=True):
        self.w = Tensor(np.random.randn(i, o) * np.sqrt(2.0 / i))
        self.b = Tensor(np.zeros(o)) if bias else None

    def forward(self, x):
        self.x = x
        return x @ self.w.data + (self.b.data if self.b else 0)

    def backward(self, dy):
        xf = self.x.reshape(-1, self.x.shape[-1])
        dyf = dy.reshape(-1, dy.shape[-1])
        self.w.grad += xf.T @ dyf
        if self.b:
            self.b.grad += dyf.sum(0)
        return (dy @ self.w.data.T).reshape(self.x.shape)

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

class SwiGLU(Module):
    def forward(self, x):
        self.x = x
        self.gate, self.val = np.split(x, 2, axis=-1)
        self.sig = 1.0 / (1.0 + np.exp(-np.clip(self.gate, -15, 15)))
        return (self.gate * self.sig) * self.val

    def backward(self, dy):
        dg = dy * self.val * self.sig * (1.0 + self.gate * (1.0 - self.sig))
        dv = dy * (self.gate * self.sig)
        return np.concatenate([dg, dv], axis=-1)

class GQA(Module):
    def __init__(self, d, h=8, g=2):
        self.d, self.h, self.g, self.hd = d, h, g, d // h
        self.wq = Linear(d, d, False)
        self.wk = Linear(d, (h // g) * self.hd, False)
        self.wv = Linear(d, (h // g) * self.hd, False)
        self.wo = Linear(d, d, False)
        self.scale = self.hd**-0.5

    def _rope(self, t, inv=False):
        b, s, h, d = t.shape
        f = 10000 ** -(np.arange(0, d, 2) / d)
        a = np.arange(s)[:, None] * f
        cos, sin = np.cos(a), np.sin(a) * (-1 if inv else 1)
        r, i = t[..., ::2], t[..., 1::2]
        out = np.empty_like(t)
        out[..., ::2] = r * cos[:, None, :] - i * sin[:, None, :]
        out[..., 1::2] = r * sin[:, None, :] + i * cos[:, None, :]
        return out

    def forward(self, x):
        b, s, _ = x.shape
        q = self.wq.forward(x).reshape(b, s, self.h, self.hd)
        k = self.wk.forward(x).reshape(b, s, self.h // self.g, self.hd)
        v = self.wv.forward(x).reshape(b, s, self.h // self.g, self.hd)
        self.qr, self.kr, self.vr = self._rope(q), self._rope(k), v
        kr_r = np.repeat(self.kr, self.g, 2)
        vr_r = np.repeat(self.vr, self.g, 2)
        att = np.einsum("bshd,bthd->bsht", self.qr, kr_r) * self.scale
        self.p = (e := np.exp(att - np.max(att, -1, keepdims=True))) / (e.sum(-1, keepdims=True) + 1e-12)
        return self.wo.forward(np.einsum("bsht,bthd->bshd", self.p, vr_r).reshape(b, s, -1))

    def backward(self, dy):
        b, s = dy.shape[:2]
        dy_wo = self.wo.backward(dy).reshape(b, s, self.h, self.hd)
        kr_r, vr_r = np.repeat(self.kr, self.g, 2), np.repeat(self.vr, self.g, 2)
        dp = np.einsum("bshd,bthd->bsht", dy_wo, vr_r)
        da = self.p * (dp - np.sum(self.p * dp, -1, keepdims=True)) * self.scale
        dqr = np.einsum("bsht,bthd->bshd", da, kr_r)
        dkr = np.einsum("bsht,bshd->bthd", da, self.qr).reshape(b, s, self.h // self.g, self.g, self.hd).sum(3)
        dvr = np.einsum("bsht,bshd->bthd", self.p, dy_wo).reshape(b, s, self.h // self.g, self.g, self.hd).sum(3)
        return self.wq.backward(self._rope(dqr, 1).reshape(b, s, -1)) + self.wk.backward(self._rope(dkr, 1).reshape(b, s, -1)) + self.wv.backward(dvr.reshape(b, s, -1))

class RedundantConsensus(Module):
    def __init__(self, d):
        self.gemini = [Linear(d, d * 2), SwiGLU(), Linear(d, d)]
        self.groq = [Linear(d, d * 2), SwiGLU(), Linear(d, d)]
        self.gate = Linear(d, 2)

    def forward(self, x):
        self.x = x
        g1 = self.gemini[0].forward(x)
        g2 = self.gemini[1].forward(g1)
        self.g_out = self.gemini[2].forward(g2)
        q1 = self.groq[0].forward(x)
        q2 = self.groq[1].forward(q1)
        self.q_out = self.groq[2].forward(q2)
        lg = self.gate.forward(x)
        self.p = (e := np.exp(lg - np.max(lg, -1, keepdims=True))) / (e.sum(-1, keepdims=True) + 1e-12)
        self.cache = (g1, g2, q1, q2)
        return self.p[..., :1] * self.g_out + self.p[..., 1:] * self.q_out

    def backward(self, dy):
        g1, g2, q1, q2 = self.cache
        dg_out = dy * self.p[..., :1]
        dq_out = dy * self.p[..., 1:]
        dp = np.stack([np.sum(dy * self.g_out, -1), np.sum(dy * self.q_out, -1)], -1)
        d_gem = self.gemini[0].backward(self.gemini[1].backward(self.gemini[2].backward(dg_out)))
        d_groq = self.groq[0].backward(self.groq[1].backward(self.groq[2].backward(dq_out)))
        d_gate = self.gate.backward(dp - np.mean(dp, -1, keepdims=True))
        return d_gem + d_groq + d_gate

class MoE(Module):
    def __init__(self, d, n=4, k=2):
        self.d, self.n, self.k = d, n, k
        self.gate = Linear(d, n)
        self.experts = [[Linear(d, d * 2, False), SwiGLU(), Linear(d, d, False)] for _ in range(n)]

    def forward(self, x):
        self.sh = x.shape
        xf = x.reshape(-1, self.d)
        lg = self.gate.forward(xf)
        p = (e := np.exp(lg - np.max(lg, -1, keepdims=True))) / (e.sum(-1, keepdims=True) + 1e-12)
        self.idx = np.argsort(p, -1)[:, -self.k:]
        self.w = np.take_along_axis(p, self.idx, -1)
        self.w /= self.w.sum(-1, keepdims=True) + 1e-12
        out, self.cache = np.zeros_like(xf), []
        for i in range(self.n):
            m = np.any(self.idx == i, -1)
            if not np.any(m):
                self.cache.append(None)
                continue
            pos = np.where(self.idx[m] == i)[1]
            h1 = self.experts[i][0].forward(xf[m])
            h2 = self.experts[i][1].forward(h1)
            h3 = self.experts[i][2].forward(h2)
            out[m] += h3 * self.w[m, pos][:, None]
            self.cache.append((m, pos, h1, h2, h3))
        return out.reshape(self.sh)

    def backward(self, dy):
        dyf = dy.reshape(-1, self.d)
        dx, dg = np.zeros((dyf.shape[0], self.d)), np.zeros((dyf.shape[0], self.n))
        for i in range(self.n):
            if self.cache[i] is None:
                continue
            m, pos, h1, h2, h3 = self.cache[i]
            dg[m, i] = np.sum(dyf[m] * h3, -1)
            dh3 = self.experts[i][2].backward(dyf[m] * self.w[m, pos][:, None])
            dh2 = self.experts[i][1].backward(dh3)
            dx[m] += self.experts[i][0].backward(dh2)
        return (dx + self.gate.backward(dg - np.mean(dg, -1, keepdims=True))).reshape(self.sh)

class SovereignBlock(Module):
    def __init__(self, d):
        self.n1, self.at = RMSNorm(d), GQA(d)
        self.n2, self.rc = RMSNorm(d), RedundantConsensus(d)
        self.n3, self.ff = RMSNorm(d), MoE(d)

    def forward(self, x):
        x = x + self.at.forward(self.n1.forward(x))
        x = x + self.rc.forward(self.n2.forward(x))
        return x + self.ff.forward(self.n3.forward(x))

    def backward(self, dy):
        dy = dy + self.ff.backward(self.n3.backward(dy))
        dy = dy + self.rc.backward(self.n2.backward(dy))
        return dy + self.at.backward(self.n1.backward(dy))

class OMEGA_ASI_V3(Module):
    def __init__(self, di, dm, do, depth=3):
        self.embed = Linear(di, dm)
        self.blocks = [SovereignBlock(dm) for _ in range(depth)]
        self.norm, self.head = RMSNorm(dm), Linear(dm, do)

    def forward(self, x):
        x = self.embed.forward(x[:, None] if x.ndim == 2 else x)
        for b in self.blocks:
            x = b.forward(x)
        return self.head.forward(self.norm.forward(x[:, -1]))

    def backward(self, dy):
        dy = self.norm.backward(self.head.backward(dy))
        db = np.zeros((dy.shape[0], self.embed.x.shape[1], dy.shape[1]))
        db[:, -1] = dy
        for b in reversed(self.blocks):
            db = b.backward(db)
        self.embed.backward(db)

class AdamW:
    def __init__(self, p, lr=1e-3, wd=0.01, b1=0.9, b2=0.999):
        self.p, self.lr, self.wd, self.b1, self.b2 = p, lr, wd, b1, b2
        self.m = [np.zeros_like(i.data) for i in p]
        self.v = [np.zeros_like(i.data) for i in p]
        self.t = 0

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
    N, D, C, BS, E = 2048, 784, 10, 128, 50
    X = np.random.randn(N, D).astype("f4")
    Y = np.random.randint(0, C, N)
    model = OMEGA_ASI_V3(D, 128, C, depth=2)
    opt = AdamW(model.params(), lr=5e-4)

    for e in range(E):
        idx = np.random.permutation(N)
        ls, ac = [], []
        for i in range(0, N, BS):
            xb, yb = X[idx[i : i + BS]], Y[idx[i : i + BS]]
            lg = model.forward(xb)
            mx = np.max(lg, -1, keepdims=True)
            pr = (p := np.exp(lg - mx)) / (p.sum(-1, keepdims=True) + 1e-12)
            loss = -np.mean(np.log(pr[range(len(yb)), yb] + 1e-12))
            acc = np.mean(pr.argmax(-1) == yb)
            ls.append(loss)
            ac.append(acc)
            dl = pr.copy()
            dl[range(len(yb)), yb] -= 1
            model.backward(dl / len(yb))
            opt.step()
        if (e + 1) % 5 == 0:
            print(f"STEP {e+1:03d} | LOSS: {np.mean(ls):.4f} | ACC: {np.mean(ac):.4f}")

if __name__ == "__main__":
    train()
