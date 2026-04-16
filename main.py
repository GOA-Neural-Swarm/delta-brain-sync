
import numpy as np

class Tensor:
    def __init__(self, data, name=""):
        self.data = data.astype("f4")
        self.grad = np.zeros_like(self.data)

class Module:
    def params(self):
        ps = []
        for v in self.__dict__.values():
            if isinstance(v, Tensor): ps.append(v)
            elif isinstance(v, Module): ps.extend(v.params())
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], Module):
                for m in v: ps.extend(m.params())
        return ps

class Linear(Module):
    def __init__(self, i, o):
        self.w = Tensor(np.random.randn(i, o) * np.sqrt(2. / i))
        self.b = Tensor(np.zeros(o))

    def forward(self, x):
        self.x = x
        return x @ self.w.data + self.b.data

    def backward(self, dy):
        self.w.grad += self.x.reshape(-1, self.x.shape[-1]).T @ dy.reshape(-1, dy.shape[-1])
        self.b.grad += dy.reshape(-1, dy.shape[-1]).sum(0)
        return dy @ self.w.data.T

class RMSNorm(Module):
    def __init__(self, d, e=1e-6):
        self.g = Tensor(np.ones(d))
        self.e = e

    def forward(self, x):
        self.x = x
        self.v = np.mean(x**2, -1, keepdims=True)
        self.inv = 1. / np.sqrt(self.v + self.e)
        self.nx = x * self.inv
        return self.g.data * self.nx

    def backward(self, dy):
        dg = dy * self.g.data
        self.g.grad += np.sum(dy * self.nx, axis=tuple(range(dy.ndim - 1)))
        return (dg - self.nx * np.mean(dg * self.nx, -1, keepdims=True)) * self.inv

class GQA(Module):
    def __init__(self, d, h=8, g=2):
        self.d, self.h, self.g, self.hd = d, h, g, d // h
        self.wq = Linear(d, d)
        self.wk = Linear(d, (h // g) * self.hd)
        self.wv = Linear(d, (h // g) * self.hd)
        self.wo = Linear(d, d)
        self.scale = self.hd**-0.5

    def _rope(self, t, inv=False):
        b, s, h, d = t.shape
        p = np.arange(s)[:, None]
        f = 10000**-(np.arange(0, d, 2) / d)
        a = p * f
        cos, sin = np.cos(a), np.sin(a)
        if inv: sin = -sin
        r, i = t[..., ::2], t[..., 1::2]
        out = np.empty_like(t)
        out[..., ::2] = r * cos[:, None, :] - i * sin[:, None, :]
        out[..., 1::2] = r * sin[:, None, :] + i * cos[:, None, : ]
        return out

    def forward(self, x):
        b, s, _ = x.shape
        q = self.wq.forward(x).reshape(b, s, self.h, self.hd)
        k = self.wk.forward(x).reshape(b, s, self.h // self.g, self.hd)
        v = self.wv.forward(x).reshape(b, s, self.h // self.g, self.hd)
        self.qr, self.kr, self.vr = self._rope(q), self._rope(k), v
        kr_rep = np.repeat(self.kr, self.g, 2)
        vr_rep = np.repeat(self.vr, self.g, 2)
        attn = np.einsum("bshd,bthd->bsht", self.qr, kr_rep) * self.scale
        self.p = np.exp(attn - np.max(attn, -1, keepdims=True))
        self.p /= (self.p.sum(-1, keepdims=True) + 1e-12)
        out = np.einsum("bsht,bthd->bshd", self.p, vr_rep).reshape(b, s, -1)
        return self.wo.forward(out)

    def backward(self, dy):
        b, s = dy.shape[:2]
        dy_wo = self.wo.backward(dy).reshape(b, s, self.h, self.hd)
        kr_rep = np.repeat(self.kr, self.g, 2)
        vr_rep = np.repeat(self.vr, self.g, 2)
        dp = np.einsum("bshd,bthd->bsht", dy_wo, vr_rep)
        da = self.p * (dp - np.sum(self.p * dp, -1, keepdims=True)) * self.scale
        dqr = np.einsum("bsht,bthd->bshd", da, kr_rep)
        dkr_f = np.einsum("bsht,bshd->bthd", da, self.qr)
        dkr = dkr_f.reshape(b, s, self.h // self.g, self.g, self.hd).sum(3)
        dvr_f = np.einsum("bsht,bshd->bthd", self.p, dy_wo)
        dvr = dvr_f.reshape(b, s, self.h // self.g, self.g, self.hd).sum(3)
        dq = self.wq.backward(self._rope(dqr, True).reshape(b, s, -1))
        dk = self.wk.backward(self._rope(dkr, True).reshape(b, s, -1))
        dv = self.wv.backward(dvr.reshape(b, s, -1))
        return dq + dk + dv

class MoE(Module):
    def __init__(self, d, n=4, k=2):
        self.d, self.n, self.k = d, n, k
        self.gate = Linear(d, n)
        self.w1 = [Linear(d, d * 2) for _ in range(n)]
        self.w2 = [Linear(d * 2, d) for _ in range(n)]

    def _swiglu(self, x):
        x, g = np.split(x, 2, -1)
        sig = 1. / (1. + np.exp(-np.clip(g, -15, 15)))
        return x * (g * sig), (x, g, sig)

    def _swiglu_back(self, dy, c):
        x, g, sig = c
        return np.concatenate([dy * (g * sig), dy * x * sig * (1. + g * (1. - sig))], -1)

    def forward(self, x):
        self.sh = x.shape
        xf = x.reshape(-1, self.d)
        logits = self.gate.forward(xf)
        p = np.exp(logits - np.max(logits, -1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        self.idx = np.argsort(p, -1)[:, -self.k:]
        self.w = np.take_along_axis(p, self.idx, -1)
        self.w /= (self.w.sum(-1, keepdims=True) + 1e-12)
        out = np.zeros_like(xf)
        self.cache = []
        for i in range(self.n):
            m = np.any(self.idx == i, axis=-1)
            if not np.any(m):
                self.cache.append(None)
                continue
            pos = np.where(self.idx[m] == i)[1]
            wi = self.w[m, pos][:, None]
            h1 = self.w1[i].forward(xf[m])
            act, c = self._swiglu(h1)
            h2 = self.w2[i].forward(act)
            out[m] += h2 * wi
            self.cache.append((m, pos, act, c, h2))
        return out.reshape(self.sh)

    def backward(self, dy):
        dyf = dy.reshape(-1, self.d)
        dx, dg = np.zeros((dyf.shape[0], self.d)), np.zeros((dyf.shape[0], self.n))
        for i in range(self.n):
            if self.cache[i] is None: continue
            m, pos, act, c, h2 = self.cache[i]
            wi = self.w[m, pos][:, None]
            dg[m, i] = np.sum(dyf[m] * h2, -1)
            dx[m] += self.w1[i].backward(self._swiglu_back(self.w2[i].backward(dyf[m] * wi), c))
        return (dx + self.gate.backward(dg - np.mean(dg, -1, keepdims=True))).reshape(self.sh)

class RedundantCompute(Module):
    def __init__(self, d):
        self.gemini = MoE(d)
        self.groq = GQA(d, h=4, g=2)
        self.alpha = Tensor(np.array([0.5]))

    def forward(self, x):
        self.og, self.oq = self.gemini.forward(x), self.groq.forward(x)
        return self.alpha.data * self.og + (1 - self.alpha.data) * self.oq

    def backward(self, dy):
        self.alpha.grad += np.sum(dy * (self.og - self.oq))
        return self.gemini.backward(dy * self.alpha.data) + self.groq.backward(dy * (1 - self.alpha.data))

class SovereignBlock(Module):
    def __init__(self, d):
        self.n1, self.at = RMSNorm(d), GQA(d)
        self.n2, self.rc = RMSNorm(d), RedundantCompute(d)

    def forward(self, x):
        x = x + self.at.forward(self.n1.forward(x))
        return x + self.rc.forward(self.n2.forward(x))

    def backward(self, dy):
        dy = dy + self.rc.backward(self.n2.backward(dy))
        return dy + self.at.backward(self.n1.backward(dy))

class OMEGA_ASI(Module):
    def __init__(self, di, dm, do, depth=2):
        self.embed = Linear(di, dm)
        self.blocks = [SovereignBlock(dm) for _ in range(depth)]
        self.fn = RMSNorm(dm)
        self.head = Linear(dm, do)

    def forward(self, x):
        x = self.embed.forward(x[:, None] if x.ndim == 2 else x)
        for b in self.blocks: x = b.forward(x)
        return self.head.forward(self.fn.forward(x[:, -1]))

    def backward(self, dy):
        dy = self.fn.backward(self.head.backward(dy))
        db = np.zeros((dy.shape[0], self.embed.x.shape[1], dy.shape[1]))
        db[:, -1] = dy
        for b in reversed(self.blocks): db = b.backward(db)
        self.embed.backward(db)

class AdamW:
    def __init__(self, p, lr=1e-3, wd=0.01, b1=0.9, b2=0.999):
        self.p, self.lr, self.wd, self.b1, self.b2 = p, lr, wd, b1, b2
        self.m = [np.zeros_like(i.data) for i in p]
        self.v = [np.zeros_like(i.data) for i in p]
        self.t = 0

    def step(self):
        self.t += 1
        lr_t = self.lr * (np.sqrt(1 - self.b2**self.t) / (1 - self.b1**self.t))
        for i, pt in enumerate(self.p):
            g = np.clip(pt.grad, -1, 1)
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g**2)
            pt.data -= lr_t * (self.m[i] / (np.sqrt(self.v[i]) + 1e-8) + self.wd * pt.data)
            pt.grad.fill(0)

def train():
    N, D, C, BS, E = 1024, 784, 10, 64, 50
    X = np.random.randn(N, D).astype("f4")
    Y = np.random.randint(0, C, N)
    model = OMEGA_ASI(D, 128, C, depth=2)
    optimizer = AdamW(model.params(), lr=1e-3, wd=0.01)

    for epoch in range(E):
        idx = np.random.permutation(N)
        ls, ac = [], []
        for i in range(0, N, BS):
            xb, yb = X[idx[i:i+BS]], Y[idx[i:i+BS]]
            logits = model.forward(xb)
            probs = np.exp(logits - np.max(logits, -1, keepdims=True))
            probs /= probs.sum(-1, keepdims=True)
            loss = -np.mean(np.log(probs[np.arange(len(yb)), yb] + 1e-12))
            acc = np.mean(np.argmax(probs, -1) == yb)
            ls.append(loss)
            ac.append(acc)
            dl = probs.copy()
            dl[np.arange(len(yb)), yb] -= 1
            model.backward(dl / len(yb))
            optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:03d} | Loss: {np.mean(ls):.4f} | Acc: {np.mean(ac):.4f}")

if __name__ == "__main__":
    train()
