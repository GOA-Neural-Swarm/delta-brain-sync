import numpy as np


class Linear:
    def __init__(self, i, o, s=None):
        self.W = np.random.randn(i, o).astype("f4") * (s if s else np.sqrt(2 / i))
        self.b = np.zeros(o, "f4")
        self.x, self.dW, self.db = None, None, None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dy):
        self.dW = self.x.reshape(-1, self.x.shape[-1]).T @ dy.reshape(-1, dy.shape[-1])
        self.db = dy.sum(axis=tuple(range(dy.ndim - 1)))
        dx = dy @ self.W.T
        return dx.reshape(self.x.shape[:-1] + (self.W.shape[0],))


class RMSNorm:
    def __init__(self, d, e=1e-6):
        self.g, self.e = np.ones(d, "f4"), e

    def forward(self, x):
        self.x = x
        self.v = np.mean(x**2, -1, keepdims=True)
        self.r = 1 / np.sqrt(self.v + self.e)
        return self.g * (x * self.r)

    def backward(self, dy):
        xn = self.x * self.r
        self.dg = (dy * xn).sum(axis=tuple(range(dy.ndim - 1)))
        dn = dy * self.g
        return self.r * (dn - xn * np.mean(dn * xn, -1, keepdims=True))


class RoPE:
    def __init__(self, d, m=4096):
        f = 1.0 / (10000 ** (np.arange(0, d, 2) / d))
        t = np.outer(np.arange(m), f)
        self.cos, self.sin = np.cos(t), np.sin(t)

    def apply(self, x, conj=False):
        s, d = x.shape[1], x.shape[-1]
        x1, x2 = x[..., : d // 2], x[..., d // 2 :]
        c, sn = self.cos[:s, None, :], self.sin[:s, None, :]
        if conj:
            return np.concatenate([x1 * c + x2 * sn, x2 * c - x1 * sn], -1)
        return np.concatenate([x1 * c - x2 * sn, x2 * c + x1 * sn], -1)


class RedundantLogicCore:
    def __init__(self, d):
        self.gemini_path = Linear(d, d * 2)
        self.groq_path = Linear(d, d * 2)
        self.out = Linear(d * 2, d)

    def forward(self, x):
        self.ge = self.gemini_path.forward(x)
        self.gr = self.groq_path.forward(x)
        # SwiGLU-style redundancy fusion
        self.act = (self.ge * (1 / (1 + np.exp(-np.clip(self.ge, -12, 12))))) * self.gr
        return self.out.forward(self.act)

    def backward(self, dy):
        dact = self.out.backward(dy)
        sig = 1 / (1 + np.exp(-np.clip(self.ge, -12, 12)))
        sw = self.ge * sig
        dge = dact * self.gr * (sig * (1 + self.ge * (1 - sig)))
        dgr = dact * sw
        return self.gemini_path.backward(dge) + self.groq_path.backward(dgr)


class SovereignGQA:
    def __init__(self, d, h=8, g=2):
        self.d, self.h, self.g, self.hd = d, h, g, d // h
        self.wq = Linear(d, d)
        self.wk = Linear(d, (h // g) * self.hd)
        self.wv = Linear(d, (h // g) * self.hd)
        self.wo = Linear(d, d)
        self.rope = RoPE(self.hd)
        self.sc = self.hd**-0.5

    def forward(self, x):
        b, s, _ = x.shape
        q = self.wq.forward(x).reshape(b, s, self.h, self.hd)
        k = self.wk.forward(x).reshape(b, s, self.h // self.g, self.hd)
        v = self.wv.forward(x).reshape(b, s, self.h // self.g, self.hd)
        self.qr, self.kr, self.v_raw = self.rope.apply(q), self.rope.apply(k), v
        ke, ve = np.repeat(self.kr, self.g, 2), np.repeat(v, self.g, 2)
        at = np.einsum("bshd,bthd->bsht", self.qr, ke) * self.sc
        at -= at.max(-1, keepdims=True)
        self.p = np.exp(at)
        self.p /= self.p.sum(-1, keepdims=True) + 1e-12
        out = np.einsum("bsht,bthd->bshd", self.p, ve).reshape(b, s, -1)
        return self.wo.forward(out)

    def backward(self, dy):
        b, s, _ = dy.shape
        dy_wo = self.wo.backward(dy).reshape(b, s, self.h, self.hd)
        ke, ve = np.repeat(self.kr, self.g, 2), np.repeat(self.v_raw, self.g, 2)
        dp = np.einsum("bshd,bthd->bsht", dy_wo, ve)
        da = self.p * (dp - (self.p * dp).sum(-1, keepdims=True)) * self.sc
        dqr = np.einsum("bsht,bthd->bshd", da, ke)
        dke = np.einsum("bsht,bshd->bthd", da, self.qr)
        dve = np.einsum("bsht,bshd->bthd", self.p, dy_wo)
        dq = self.rope.apply(dqr, True)
        dk = self.rope.apply(
            dke.reshape(b, s, self.h // self.g, self.g, self.hd).sum(3), True
        )
        dv = dve.reshape(b, s, self.h // self.g, self.g, self.hd).sum(3)
        return (
            self.wq.backward(dq.reshape(b, s, -1))
            + self.wk.backward(dk.reshape(b, s, -1))
            + self.wv.backward(dv.reshape(b, s, -1))
        )


class SovereignMoE:
    def __init__(self, d, n=4, e=2):
        self.n, self.d, self.f = n, d, d * e
        self.gate = Linear(d, n)
        self.experts = [
            [Linear(d, self.f), Linear(self.f, d), Linear(d, self.f)] for _ in range(n)
        ]

    def forward(self, x):
        self.x = x
        g_lg = self.gate.forward(x)
        g_lg -= g_lg.max(-1, keepdims=True)
        self.pr = np.exp(g_lg)
        self.pr /= self.pr.sum(-1, keepdims=True)
        self.cache, out = [], np.zeros_like(x)
        for i in range(self.n):
            w1, w2, w3 = self.experts[i]
            x1, x3 = w1.forward(x), w3.forward(x)
            sig = 1 / (1 + np.exp(-np.clip(x1, -12, 12)))
            sw = x1 * sig
            act = sw * x3
            o = self.w2[i].forward(act)
            self.cache.append((x1, x3, sig, sw, act))
            out += self.pr[..., i : i + 1] * o
        return out

    def backward(self, dy):
        dx, dpr = np.zeros_like(self.x), np.zeros_like(self.pr)
        for i in range(self.n):
            x1, x3, sig, sw, act = self.cache[i]
            dpr[..., i] = (dy * self.w2[i].forward(act)).sum(-1)
            dy_exp = dy * self.pr[..., i : i + 1]
            dact = self.w2[i].backward(dy_exp)
            dx3, dsw = dact * sw, dact * x3
            dx1 = dsw * (sig * (1 + x1 * (1 - sig)))
            dx += w1.backward(dx1) + w3.backward(dx3)
        dl = self.pr * (dpr - (self.pr * dpr).sum(-1, keepdims=True))
        return dx + self.gate.backward(dl)


class SovereignBlock:
    def __init__(self, d):
        self.ln1, self.attn = RMSNorm(d), SovereignGQA(d)
        self.ln2, self.logic = RMSNorm(d), RedundantLogicCore(d)
        self.ln3, self.moe = RMSNorm(d), SovereignMoE(d)

    def forward(self, x):
        x = x + self.attn.forward(self.ln1.forward(x))
        x = x + self.logic.forward(self.ln2.forward(x))
        x = x + self.moe.forward(self.ln3.forward(x))
        return x

    def backward(self, dy):
        dy = dy + self.ln3.backward(self.moe.backward(dy))
        dy = dy + self.ln2.backward(self.logic.backward(dy))
        dy = dy + self.ln1.backward(self.attn.backward(dy))
        return dy


class OMEGA_ASI:
    def __init__(self, i, h, o, d=2):
        self.emb = Linear(i, h)
        self.blks = [SovereignBlock(h) for _ in range(d)]
        self.ln = RMSNorm(h)
        self.head = Linear(h, o)

    def forward(self, x):
        if x.ndim == 2:
            x = x[:, None]
        x = self.emb.forward(x)
        for b in self.blks:
            x = b.forward(x)
        return self.head.forward(self.ln.forward(x[:, -1]))

    def backward(self, dy):
        dy = self.ln.backward(self.head.backward(dy))
        dys = np.zeros((dy.shape[0], 1, dy.shape[1]), "f4")
        dys[:, -1] = dy
        for b in reversed(self.blks):
            dys = b.backward(dys)
        self.emb.backward(dys)

    def params(self):
        p = []

        def g(obj):
            if isinstance(obj, (Linear, RMSNorm)):
                p.append(obj)
            elif isinstance(obj, list):
                [g(i) for i in obj]
            elif hasattr(obj, "__dict__"):
                [g(v) for k, v in obj.__dict__.items() if k[0] != "_"]

        g(self)
        return list(set(p))


class AdamW:
    def __init__(self, p, lr=1e-3, b1=0.9, b2=0.999, wd=0.01):
        self.p, self.lr, self.b1, self.b2, self.wd, self.t = p, lr, b1, b2, wd, 0
        self.m = {
            id(x): [
                np.zeros_like(getattr(x, a))
                for a in (["W", "b"] if hasattr(x, "W") else ["g"])
            ]
            for x in p
        }
        self.v = {
            id(x): [
                np.zeros_like(getattr(x, a))
                for a in (["W", "b"] if hasattr(x, "W") else ["g"])
            ]
            for x in p
        }

    def step(self):
        self.t += 1
        lr_t = (
            self.lr
            * min(1.0, self.t / 100)
            * (0.5 * (1 + np.cos(min(self.t, 1000) * np.pi / 1000)))
        )
        for x in self.p:
            at = ["W", "b"] if hasattr(x, "W") else ["g"]
            for i, a in enumerate(at):
                gr = getattr(x, "d" + a if a != "g" else "dg")
                m, v = self.m[id(x)][i], self.v[id(x)][i]
                m[:] = self.b1 * m + (1 - self.b1) * gr
                v[:] = self.b2 * v + (1 - self.b2) * (gr**2)
                mh, vh = m / (1 - self.b1**self.t), v / (1 - self.b2**self.t)
                pv = getattr(x, a)
                pv -= lr_t * (mh / (np.sqrt(vh) + 1e-8) + self.wd * pv)
                setattr(x, a, pv)


def train():
    N, D, C, BS, E = 1024, 784, 10, 32, 100
    X = np.random.randn(N, D).astype("f4")
    Y = np.random.randint(0, C, N)
    model = OMEGA_ASI(D, 128, C, d=2)
    ps = model.params()
    opt = AdamW(ps, lr=2e-3)

    for epoch in range(E):
        idx = np.random.permutation(N)
        l_sum, a_sum = 0, 0
        for i in range(0, N, BS):
            xb, yb = X[idx[i : i + BS]], Y[idx[i : i + BS]]
            logits = model.forward(xb)
            probs = np.exp(logits - logits.max(-1, keepdims=True))
            probs /= probs.sum(-1, keepdims=True)
            l_sum += -np.log(probs[range(len(yb)), yb] + 1e-12).sum()
            a_sum += (probs.argmax(1) == yb).sum()
            dy = probs.copy()
            dy[range(len(yb)), yb] -= 1
            model.backward(dy / len(yb))

            gn = np.sqrt(
                sum(
                    (getattr(p, "d" + a if a != "g" else "dg") ** 2).sum()
                    for p in ps
                    for a in (["W", "b"] if hasattr(p, "W") else ["g"])
                )
            )
            if gn > 1.0:
                for p in ps:
                    for a in (["W", "b"] if hasattr(p, "W") else ["g"]):
                        k = "d" + a if a != "g" else "dg"
                        setattr(p, k, getattr(p, k) / gn)
            opt.step()
        if (epoch + 1) % 5 == 0:
            print(
                f"EPOCH {epoch+1:03d} | LOSS: {l_sum/N:.4f} | ACC: {a_sum/N:.4f} | GRAD: {gn:.2f}"
            )


if __name__ == "__main__":
    train()
