
import numpy as np

class Tensor:
    def __init__(self, data, name=""):
        self.data = data.astype("f4")
        self.grad = np.zeros_like(self.data)
        self.m = np.zeros_like(self.data)
        self.v = np.zeros_like(self.data)

class Module:
    def params(self):
        ps = []
        for v in self.__dict__.values():
            if isinstance(v, Tensor): ps.append(v)
            elif isinstance(v, Module): ps.extend(v.params())
            elif isinstance(v, list):
                for i in v:
                    if isinstance(i, Module): ps.extend(i.params())
        return ps

class Linear(Module):
    def __init__(self, i, o, scale=1.0):
        self.w = Tensor(np.random.randn(i, o) * (np.sqrt(2.0 / i) * scale))
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
        self.inv = 1.0 / np.sqrt(self.v + self.e)
        self.nx = x * self.inv
        return self.g.data * self.nx

    def backward(self, dy):
        dg = dy * self.g.data
        self.g.grad += np.sum(dy * self.nx, axis=tuple(range(dy.ndim - 1)))
        return (dg - self.nx * np.mean(dg * self.nx, -1, keepdims=True)) * self.inv

class GeminiGroq(Module):
    def __init__(self, d, h=8, g=2, k=2):
        self.d, self.h, self.g, self.k, self.hd = d, h, g, k, d // h
        self.wq = Linear(d, d)
        self.wk = Linear(d, (h // g) * self.hd)
        self.wv = Linear(d, (h // g) * self.hd)
        self.wo = Linear(d, d)
        self.scale = self.hd**-0.5
        self.gate = Linear(d, h // g)
        self.w1 = [Linear(d, d * 2) for _ in range(h // g)]
        self.w2 = [Linear(d * 2, d) for _ in range(h // g)]

    def _rope(self, t, inv=False):
        b, s, h, d = t.shape
        p = np.arange(s)[:, None]
        f = 10000 ** -(np.arange(0, d, 2) / d)
        a = p * f
        cos, sin = np.cos(a), np.sin(a)
        if inv: sin = -sin
        r, i = t[..., ::2], t[..., 1::2]
        out = np.empty_like(t)
        out[..., ::2] = r * cos[:, None, :] - i * sin[:, None, :]
        out[..., 1::2] = r * sin[:, None, :] + i * cos[:, None, :]
        return out

    def _swiglu(self, x):
        x, g = np.split(x, 2, -1)
        sig = 1.0 / (1.0 + np.exp(-np.clip(g, -15, 15)))
        return x * (g * sig), (x, g, sig)

    def _swiglu_back(self, dy, c):
        x, g, sig = c
        return np.concatenate([dy * (g * sig), dy * x * sig * (1.0 + g * (1.0 - sig))], -1)

    def forward(self, x):
        b, s, _ = x.shape
        q = self.wq.forward(x).reshape(b, s, self.h, self.hd)
        k = self.wk.forward(x).reshape(b, s, self.h // self.g, self.hd)
        v = self.wv.forward(x).reshape(b, s, self.h // self.g, self.hd)
        qr, kr, vr = self._rope(q), self._rope(k), v
        kr_rep = np.repeat(kr, self.g, 2)
        vr_rep = np.repeat(vr, self.g, 2)
        attn = np.einsum("bshd,bthd->bsht", qr, kr_rep) * self.scale
        logits = self.gate.forward(x).reshape(b, s, self.h // self.g)
        p = np.exp(logits - np.max(logits, -1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        idx = np.argsort(p, -1)[:, :, -self.k:]
        w = np.take_along_axis(p, idx, -1)
        w /= w.sum(-1, keepdims=True) + 1e-12
        out = np.zeros_like(x)
        cache = []
        for i in range(self.h // self.g):
            m = np.any(idx == i, axis=-1)
            if not np.any(m):
                cache.append(None)
                continue
            pos = np.where(idx[m] == i)[1]
            wi = w[m, pos][:, None]
            h1 = self.w1[i].forward(x[m])
            act, c = self._swiglu(h1)
            h2 = self.w2[i].forward(act)
            out[m] += h2 * wi
            cache.append((m, pos, act, c, h2))
        out += np.einsum("bshd,bthd->bshd", np.einsum("bsht,bthd->bshd", np.exp(attn - np.max(attn, -1, keepdims=True)), vr_rep), qr)
        out /= out.sum(-1, keepdims=True) + 1e-12
        return self.wo.forward(out)

    def backward(self, dy):
        dy_wo = self.wo.backward(dy).reshape(dy.shape[0], dy.shape[1], self.h, self.hd)
        kr_rep = np.repeat(self._rope(self.wk.forward(dy)).reshape(dy.shape[0], dy.shape[1], self.h // self.g, self.hd), self.g, 2)
        vr_rep = np.repeat(self.wv.forward(dy).reshape(dy.shape[0], dy.shape[1], self.h // self.g, self.hd), self.g, 2)
        dp = np.einsum("bshd,bthd->bsht", dy_wo, vr_rep)
        da = np.exp(dp - np.max(dp, -1, keepdims=True)) * self.scale
        dqr = np.einsum("bsht,bthd->bshd", da, kr_rep)
        dkr_f = np.einsum("bsht,bshd->bthd", da, self._rope(self.wq.forward(dy)).reshape(dy.shape[0], dy.shape[1], self.h, self.hd))
        dkr = dkr_f.reshape(dy.shape[0], dy.shape[1], self.h // self.g, self.g, self.hd).sum(3)
        dvr = (np.einsum("bsht,bshd->bthd", np.exp(dp - np.max(dp, -1, keepdims=True)), dy_wo).reshape(dy.shape[0], dy.shape[1], self.h // self.g, self.g, self.hd).sum(3))
        dq = self.wq.backward(self._rope(dqr, True).reshape(dy.shape[0], dy.shape[1], -1))
        dk = self.wk.backward(self._rope(dkr, True).reshape(dy.shape[0], dy.shape[1], -1))
        dv = self.wv.backward(dvr.reshape(dy.shape[0], dy.shape[1], -1))
        return dq + dk + dv

class SovereignBlock(Module):
    def __init__(self, d):
        self.n1, self.at = RMSNorm(d), GeminiGroq(d)

    def forward(self, x):
        x = x + self.at.forward(self.n1.forward(x))
        return x

    def backward(self, dy):
        dy_at = self.at.backward(self.n1.backward(dy))
        return dy + dy_at

class OMEGA_ASI(Module):
    def __init__(self, di, dm, do, depth=2):
        self.embed = Linear(di, dm)
        self.blocks = [SovereignBlock(dm) for _ in range(depth)]
        self.fn = RMSNorm(dm)
        self.head = Linear(dm, do)

    def forward(self, x):
        if x.ndim == 2: x = x[:, None, :]
        x = self.embed.forward(x)
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
        self.t = 0

    def step(self):
        self.t += 1
        lr_t = self.lr * (np.sqrt(1 - self.b2**self.t) / (1 - self.b1**self.t))
        for pt in self.p:
            g = np.clip(pt.grad, -1.0, 1.0)
            pt.m = self.b1 * pt.m + (1 - self.b1) * g
            pt.v = self.b2 * pt.v + (1 - self.b2) * (g**2)
            pt.data -= lr_t * (pt.m / (np.sqrt(pt.v) + 1e-8) + self.wd * pt.data)
            pt.grad.fill(0)

def train():
    N, D, C, BS, E = 1024, 784, 10, 64, 50
    X = np.random.randn(N, D).astype("f4")
    Y = np.random.randint(0, C, N)
    model = OMEGA_ASI(D, 128, C, depth=2)
    optimizer = AdamW(model.params(), lr=2e-3, wd=0.05)

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
            ls.append(loss); ac.append(acc)
            dl = probs.copy()
            dl[np.arange(len(yb)), yb] -= 1
            model.backward(dl / len(yb))
            optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:03d} | Loss: {np.mean(ls):.4f} | Acc: {np.mean(ac):.4f}")

if __name__ == "__main__":
    train()
