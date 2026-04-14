import numpy as np


class Activation:
    @staticmethod
    def swiglu(x):
        h = x.shape[-1] // 2
        a, b = x[..., :h], x[..., h:]
        return a * (1 / (1 + np.exp(-np.clip(a, -12, 12)))) * b

    @staticmethod
    def d_swiglu(x, d):
        h = x.shape[-1] // 2
        a, b = x[..., :h], x[..., h:]
        s = 1 / (1 + np.exp(-np.clip(a, -12, 12)))
        sw = a * s
        da = d * b * (s + sw * (1 - s))
        db = d * sw
        return np.concatenate([da, db], -1)

    @staticmethod
    def geglu(x):
        h = x.shape[-1] // 2
        a, b = x[..., :h], x[..., h:]
        g = 0.5 * a * (1 + np.tanh(0.79788 * (a + 0.0447 * a**3)))
        return g * b

    @staticmethod
    def d_geglu(x, d):
        h = x.shape[-1] // 2
        a, b = x[..., :h], x[..., h:]
        t = np.tanh(0.79788 * (a + 0.0447 * a**3))
        dg = 0.5 * (1 + t) + a * (0.39894 * np.exp(-0.5 * a**2))
        da = d * b * dg
        db = d * (0.5 * a * (1 + t))
        return np.concatenate([da, db], -1)

    @staticmethod
    def softmax(x, axis=-1):
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e / (np.sum(e, axis=axis, keepdims=True) + 1e-10)


class Linear:
    def __init__(self, i, o, std=None):
        self.W = np.random.randn(i, o).astype("f") * (std or (2/i)**0.5)
        self.b = np.zeros(o, "f")
        self.dW, self.db = np.zeros_like(self.W), np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, d):
        self.dW = self.x.reshape(-1, self.x.shape[-1]).T @ d.reshape(-1, d.shape[-1])
        self.db = d.sum(axis=tuple(range(d.ndim - 1)))
        return d @ self.W.T


class RMSNorm:
    def __init__(self, d, e=1e-6):
        self.g = np.ones(d, "f")
        self.e = e

    def forward(self, x):
        self.x = x
        self.v = 1 / (np.mean(x**2, -1, keepdims=True) + self.e)**0.5
        return self.g * (x * self.v)

    def backward(self, d):
        nx = self.x * self.v
        self.dg = (d * nx).sum(axis=tuple(range(d.ndim - 1)))
        dn = d * self.g
        return self.v * (dn - nx * np.mean(dn * nx, -1, keepdims=True))


class RoPE:
    def __init__(self, d, m=4096):
        f = 1 / (10000 ** (np.arange(0, d, 2) / d))
        fr = np.outer(np.arange(m), f)
        self.c, self.s = np.cos(fr), np.sin(fr)

    def apply(self, x, rev=False):
        s, d2 = x.shape[1], x.shape[-1] // 2
        c, n = self.c[:s][None, :, None, :], self.s[:s][None, :, None, :]
        x1, x2 = x[..., :d2], x[..., d2:]
        c, sn = self.c[:s][None, :, None, :], self.s[:s][None, :, None, :]
        if rev:
            return np.concatenate([x1 * c + x2 * sn, x2 * c - x1 * sn], -1)
        return np.concatenate([x1 * c - x2 * sn, x2 * c + x1 * sn], -1)


class GQA:
    def __init__(self, d, h=8, k=2):
        self.d, self.h, self.k, self.hd = d, h, k, d // h
        self.g = h // k
        self.wq, self.wk, self.wv, self.wo = (
            Linear(d, d),
            Linear(d, k * self.hd),
            Linear(d, k * self.hd),
            Linear(d, d),
        )
        self.rope, self.sc = RoPE(self.hd), self.hd**-0.5
        self.q, self.k_cache, self.v_cache, self.p = None, None, None, None

    def forward(self, x):
        b, s, _ = x.shape
        q = self.wq.forward(x).reshape(b, s, self.h, self.hd)
        k = self.wk.forward(x).reshape(b, s, self.k, self.hd)
        v = self.wv.forward(x).reshape(b, s, self.k, self.hd)
        self.q, self.k_c, self.v_c = self.rope.apply(q), self.rope.apply(k), v
        kr, vr = np.repeat(self.k_c, self.g, 2), np.repeat(v, self.g, 2)
        at = np.einsum("bshd,bthd->bsht", self.q, kr) * self.sc
        self.p = Ops.softmax(at)
        return self.wo.forward(np.einsum("bsht,bthd->bshd", self.p, vr).reshape(b, s, self.d))

    def backward(self, d):
        b, s, _ = d.shape
        do = self.wo.backward(d).reshape(b, s, self.h, self.hd)
        kr, vr = np.repeat(self.k_c, self.g, 2), np.repeat(self.v_c, self.g, 2)
        dv_full = np.einsum("bsht,bshd->bthd", self.p, do)
        dp = np.einsum("bshd,bthd->bsht", do, vr)
        da = self.p * (dp - (self.p * dp).sum(-1, keepdims=True)) * self.sc
        dq = self.rope.apply(np.einsum("bsht,bthd->bshd", da, kr), True)
        dkr = np.einsum("bsht,bshd->bthd", da, self.q)
        dk = self.rope.apply(dkr.reshape(b, s, self.k, self.g, self.hd).sum(3), True)
        dv = dvr.reshape(b, s, self.k, self.g, self.hd).sum(3)
        return (
            self.wq.backward(dq.reshape(b, s, -1))
            + self.wk.backward(dk.reshape(b, s, -1))
            + self.wv.backward(dv.reshape(b, s, -1))
        )


class RedundantLogic:
    def __init__(self, d, n_experts=4):
        self.d, self.n = d, n_experts
        self.gate = Linear(d, n_experts + 1)
        self.groq_stream = [Linear(d, d * 2), Linear(d, d)]
        self.gemini_experts = [[Linear(d, d * 2), Linear(d, d)] for _ in range(n_experts)]

    def forward(self, x):
        self.h_gem = Activation.swiglu(self.gemini[0].forward(x))
        self.o_gem = self.gemini[1].forward(self.h_gem)
        self.h_gro = Activation.gelu(self.groq[0].forward(x))
        self.o_gro = self.groq[1].forward(self.h_gro)
        g = self.gate.forward(x)
        ex = np.exp(g - g.max(-1, keepdims=True))
        self.p = ex / (ex.sum(-1, keepdims=True) + 1e-10)
        return self.p[..., :1] * self.o_gem + self.p[..., 1:2] * self.o_gro

    def backward(self, d):
        dp = np.stack([(d * self.o_gem).sum(-1), (d * self.o_gro).sum(-1)], -1)
        dg = self.p * (dp - (self.p * dp).sum(-1, keepdims=True))
        dx = self.gate.backward(dg)
        d_gem = self.gemini[1].backward(d * self.p[..., :1])
        dx += self.gemini[0].backward(Activation.swiglu_grad(self.gemini[0].x, d_gem))
        d_gro = self.groq[1].backward(d * self.p[..., 1:2])
        dx += self.groq[0].backward(Activation.gelu_grad(self.groq[0].x, d_gro))
        return dx


class SparseMoE:
    def __init__(self, d, n=4, k=2):
        self.d, self.n, self.k = d, n, k
        self.gate = Linear(d, n)
        self.experts = [[Linear(d, d * 2), Linear(d, d)] for _ in range(n)]
        self.p, self.indices, self.cache = None, None, []

    def forward(self, x):
        s = x.shape
        xf = x.reshape(-1, self.d)
        logits = self.gate.forward(xf)
        ex = np.exp(logits - logits.max(-1, keepdims=True))
        probs = ex / (ex.sum(-1, keepdims=True) + 1e-10)
        self.indices = np.argsort(probs, axis=-1)[:, -self.k :]
        self.p = np.take_along_axis(probs, self.indices, axis=-1)
        self.p /= self.p.sum(-1, keepdims=True) + 1e-10
        out = np.zeros_like(xf)
        self.cache = []
        for i in range(self.n):
            pi = self.p[:, i+1:i+2]
            h = Ops.swiglu(self.gemini_experts[i][0].forward(xf))
            y = self.gemini_experts[i][1].forward(h)
            out += y * pi
            self.expert_data.append((h, y, pi))
            
        return out.reshape(s)

    def backward(self, d):
        df = d.reshape(-1, self.d)
        dx = np.zeros_like(df)
        dp = np.zeros_like(self.p)
        
        dp[:, :1] = (df * self.o_groq).sum(-1, keepdims=True)
        dx_groq = self.groq_stream[0].backward(Ops.d_geglu(self.groq_stream[0].x, self.groq_stream[1].backward(df * self.p[:, :1])))
        dx += dx_groq
        
        for i in range(self.n):
            if self.cache[i] is None:
                continue
            mask, p_idx, h, y = self.cache[i]
            ep = self.p[mask, p_idx, None]
            dpf[mask, i] = (df[mask] * y).sum(-1)
            dy = df[mask] * ep
            dh = self.experts[i][1].backward(dy)
            dx[mask] += self.experts[i][0].backward(
                Activation.swiglu_grad(self.experts[i][0].x, dh)
            )
        lp = np.zeros((df.shape[0], self.n))
        np.put_along_axis(lp, self.indices, self.p, axis=-1)
        dg = lp * (dpf - (lp * dpf).sum(-1, keepdims=True))
        return (dx + self.gate.backward(dg)).reshape(s)


class Block:
    def __init__(self, d):
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn = GQA(d)
        self.logic = RedundantLogic(d)

    def forward(self, x):
        x = x + self.attn.forward(self.n1.forward(x))
        x = x + self.logic.forward(self.n2.forward(x))
        return x

    def backward(self, d):
        d = d + self.n2.backward(self.logic.backward(d))
        d = d + self.n1.backward(self.attn.backward(d))
        return d


class OMEGA_ASI:
    def __init__(self, i=784, h=128, o=10, depth=2):
        self.embed = Linear(i, h)
        self.blocks = [Block(h) for _ in range(depth)]
        self.final_norm = RMSNorm(h)
        self.head = Linear(h, o)

    def forward(self, x):
        x = self.stem.forward(x)[:, None, :]
        for b in self.blocks:
            x = b.forward(x)
        self.feat = self.norm.forward(x[:, 0, :])
        return self.head.forward(self.feat)

    def backward(self, d):
        d = self.norm.backward(self.head.backward(d))[:, None, :]
        for b in reversed(self.blocks):
            d = b.backward(d)
        self.stem.backward(d[:, 0, :])

    def get_params(self):
        p = []

        def find(obj):
            if isinstance(obj, (Linear, RMSNorm)):
                p.append(obj)
            elif isinstance(obj, list):
                [find(i) for i in obj]
            elif hasattr(obj, "__dict__"):
                [find(v) for v in obj.__dict__.values()]

        find(self)
        return params


class Lion:
    def __init__(self, params, lr=1e-4, b1=0.9, b2=0.99, wd=0.01):
        self.params, self.lr, self.b1, self.b2, self.wd = params, lr, b1, b2, wd
        self.m = [np.zeros_like(p.W if hasattr(p, "W") else p.g) for p in params]
        self.mb = [np.zeros_like(p.b) if hasattr(p, "b") else None for p in params]

    def step(self):
        for i, p in enumerate(self.params):
            if hasattr(p, "W"):
                for attr, mom in [("W", self.m), ("b", self.mb)]:
                    if mom[i] is None:
                        continue
                    g, w = getattr(p, "d" + attr), getattr(p, attr)
                    u = np.sign(self.b1 * mom[i] + (1 - self.b1) * g)
                    w -= self.lr * (u + self.wd * w if attr == "W" else u)
                    mom[i] = self.b2 * mom[i] + (1 - self.b2) * g
                    setattr(p, attr, w)
            else:
                u = np.sign(self.b1 * self.m[i] + (1 - self.b1) * p.dg)
                p.g -= self.lr * (u + self.wd * p.g)
                self.m[i] = self.b2 * self.m[i] + (1 - self.b2) * p.dg


def train():
    N, D, C = 4096, 784, 10
    X = np.random.randn(N, D).astype("f")
    Y = np.random.randint(0, C, N)
    
    model = OMEGA_ASI(D, 128, C, 2)
    params = model.get_params()
    optimizer = Lion(params, lr=3e-4)
    
    batch_size = 64
    for epoch in range(50):
        idx = np.random.permutation(N)
        t_loss, t_acc = 0, 0
        for i in range(0, N, 64):
            xb, yb = X[idx[i : i + 64]], Y[idx[i : i + 64]]
            logits = model.forward(xb)
            ex = np.exp(logits - logits.max(1, keepdims=True))
            probs = ex / (ex.sum(1, keepdims=True) + 1e-10)
            t_loss += -np.log(probs[range(len(yb)), yb] + 1e-10).mean() * len(yb)
            t_acc += (probs.argmax(1) == yb).sum()
            dout = probs.copy()
            dout[range(len(yb)), yb] -= 1
            model.backward(dout / len(yb))

            gn = np.sqrt(
                sum(
                    (getattr(p, "dW", 0) ** 2).sum()
                    + (getattr(p, "db", 0) ** 2).sum()
                    + (getattr(p, "dg", 0) ** 2).sum()
                    for p in params
                )
            )
            if gn > 1.0:
                for p in params:
                    if hasattr(p, "dW"):
                        p.dW /= gn
                        p.db /= gn
                    if hasattr(p, "dg"):
                        p.dg /= gn
            optimizer.step()
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:02d} | Loss: {total_loss/N:.4f} | Acc: {total_acc/N:.4f}")


if __name__ == "__main__":
    train()
