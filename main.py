
import numpy as np

class FastOps:
    @staticmethod
    def silu(x):
        return x / (1.0 + np.exp(-np.clip(x, -14, 14)))

    @staticmethod
    def d_silu(x, d):
        s = 1.0 / (1.0 + np.exp(-np.clip(x, -14, 14)))
        return d * (s * (1.0 + x * (1.0 - s)))

    @staticmethod
    def swiglu_fwd(x1, x2):
        return FastOps.silu(x1) * x2

    @staticmethod
    def swiglu_bwd(x1, x2, d):
        return FastOps.d_silu(x1, d * x2), d * FastOps.silu(x1)

    @staticmethod
    def softmax(x):
        m = np.max(x, axis=-1, keepdims=True)
        e = np.exp(x - m)
        return e / (np.sum(e, axis=-1, keepdims=True) + 1e-12)

class Linear:
    def __init__(self, i, o, name=""):
        self.W = (np.random.randn(i, o) * np.sqrt(2.0 / i)).astype("f")
        self.b = np.zeros(o, "f")
        self.x, self.dW, self.db = None, None, None

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
        self.e, self.x, self.v = e, None, None

    def forward(self, x):
        self.x = x
        self.v = 1.0 / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.e)
        return self.g * (x * self.v)

    def backward(self, d):
        nx = self.x * self.v
        self.dg = (d * nx).sum(axis=tuple(range(d.ndim - 1)))
        dn = d * self.g
        return self.v * (dn - nx * np.mean(dn * nx, axis=-1, keepdims=True))

class RoPE:
    def __init__(self, d, m=2048):
        f = 1.0 / (10000 ** (np.arange(0, d, 2) / d))
        t = np.arange(m)
        fr = np.outer(t, f)
        self.cos, self.sin = np.cos(fr), np.sin(fr)

    def apply(self, x, rev=False):
        b, s, h, d = x.shape
        d2 = d // 2
        c, sn = self.cos[:s][None, :, None, :], self.sin[:s][None, :, None, :]
        x1, x2 = x[..., :d2], x[..., d2:]
        if rev: return np.concatenate([x1 * c + x2 * sn, x2 * c - x1 * sn], axis=-1)
        return np.concatenate([x1 * c - x2 * sn, x2 * c + x1 * sn], axis=-1)

class SovereignAttention:
    def __init__(self, d, h=8, k=4):
        self.d, self.h, self.k, self.hd = d, h, k, d // h
        self.g = h // k
        self.wq, self.wk, self.wv, self.wo = Linear(d, d), Linear(d, k * self.hd), Linear(d, k * self.hd), Linear(d, d)
        self.rope, self.sc = RoPE(self.hd), self.hd**-0.5

    def forward(self, x):
        b, s, _ = x.shape
        q = self.wq.forward(x).reshape(b, s, self.h, self.hd)
        k = self.wk.forward(x).reshape(b, s, self.k, self.hd)
        v = self.wv.forward(x).reshape(b, s, self.k, self.hd)
        self.q, self.kc, self.vc = self.rope.apply(q), self.rope.apply(k), v
        kr, vr = np.repeat(self.kc, self.g, axis=2), np.repeat(self.vc, self.g, axis=2)
        at = np.einsum("bshd,bthd->bsht", self.q, kr) * self.sc
        self.p = FastOps.softmax(at)
        out = np.einsum("bsht,bthd->bshd", self.p, vr).reshape(b, s, self.d)
        return self.wo.forward(out)

    def backward(self, d):
        b, s, _ = d.shape
        do = self.wo.backward(d).reshape(b, s, self.h, self.hd)
        kr, vr = np.repeat(self.kc, self.g, axis=2), np.repeat(self.vc, self.g, axis=2)
        dvr = np.einsum("bsht,bshd->bthd", self.p, do)
        dp = np.einsum("bshd,bthd->bsht", do, vr)
        da = self.p * (dp - np.sum(self.p * dp, axis=-1, keepdims=True)) * self.sc
        dq = self.rope.apply(np.einsum("bsht,bthd->bshd", da, kr), rev=True)
        dkr = np.einsum("bsht,bshd->bthd", da, self.q)
        dkc = self.rope.apply(dkr.reshape(b, s, self.k, self.g, self.hd).sum(axis=3), rev=True)
        dvc = dvr.reshape(b, s, self.k, self.g, self.hd).sum(axis=3)
        return self.wq.backward(dq.reshape(b, s, -1)) + self.wk.backward(dkc.reshape(b, s, -1)) + self.wv.backward(dvc.reshape(b, s, -1))

class SovereignMLP:
    def __init__(self, d, exp=4):
        self.w1, self.w2, self.w3 = Linear(d, d * exp), Linear(d, d * exp), Linear(d * exp, d)

    def forward(self, x):
        self.x1, self.x2 = self.w1.forward(x), self.w2.forward(x)
        self.act = FastOps.swiglu_fwd(self.x1, self.x2)
        return self.w3.forward(self.act)

    def backward(self, d):
        d3 = self.w3.backward(d)
        ds1, ds2 = FastOps.swiglu_bwd(self.x1, self.x2, d3)
        return self.w1.backward(ds1) + self.w2.backward(ds2)

class GeminiGroqBlock:
    def __init__(self, d):
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.gemini_path = SovereignAttention(d)
        self.groq_path = SovereignMLP(d)
        self.gate = Linear(d, 2)

    def forward(self, x):
        self.res = x
        nx = self.n1.forward(x)
        self.g_logits = self.gate.forward(nx)
        self.g_probs = FastOps.softmax(self.g_logits)
        self.out_gemini = self.gemini_path.forward(nx)
        self.out_groq = self.groq_path.forward(self.n2.forward(nx))
        combined = (self.g_probs[..., 0:1] * self.out_gemini) + (self.g_probs[..., 1:2] * self.out_groq)
        return self.res + combined

    def backward(self, d):
        nx = self.n1.x
        dg_gemini = d * self.g_probs[..., 0:1]
        dg_groq = d * self.g_probs[..., 1:2]
        d_gemini = self.gemini_path.backward(dg_gemini)
        d_groq = self.n2.backward(self.groq_path.backward(dg_groq))
        d_logits_gemini = (d * self.out_gemini).sum(axis=-1, keepdims=True)
        d_logits_groq = (d * self.out_groq).sum(axis=-1, keepdims=True)
        d_logits = np.concatenate([d_logits_gemini, d_logits_groq], axis=-1)
        d_gate = self.gate.backward(self.g_probs * (d_logits - np.sum(self.g_probs * d_logits, axis=-1, keepdims=True)))
        return d + self.n1.backward(d_gemini + d_groq + d_gate)

class OMEGA_ASI:
    def __init__(self, i=784, h=256, o=10, depth=4):
        self.stem = Linear(i, h)
        self.blocks = [GeminiGroqBlock(h) for _ in range(depth)]
        self.norm = RMSNorm(h)
        self.head = Linear(h, o)

    def forward(self, x):
        if x.ndim == 2: x = x[:, None, :]
        x = self.stem.forward(x)
        for b in self.blocks: x = b.forward(x)
        self.feat = self.norm.forward(x[:, 0, :])
        return self.head.forward(self.feat)

    def backward(self, d):
        d = self.norm.backward(self.head.backward(d))[:, None, :]
        for b in reversed(self.blocks): d = b.backward(d)
        self.stem.backward(d[:, 0, :])

    def get_params(self):
        p = []
        def find(obj):
            if isinstance(obj, (Linear, RMSNorm)): p.append(obj)
            elif isinstance(obj, list): [find(i) for i in obj]
            elif hasattr(obj, "__dict__"): [find(v) for v in obj.__dict__.values() if v is not obj and not isinstance(v, (np.ndarray, float, int, str))]
        find(self)
        return list(set(p))

class Lion:
    def __init__(self, params, lr=1e-4, b1=0.9, b2=0.99, wd=0.01):
        self.params, self.lr, self.b1, self.b2, self.wd = params, lr, b1, b2, wd
        self.m = {id(p): np.zeros_like(p.W) if hasattr(p, "W") else np.zeros_like(p.g) for p in params}
        self.mb = {id(p): np.zeros_like(p.b) if hasattr(p, "b") else None for p in params}

    def step(self, lr_scale=1.0):
        lr = self.lr * lr_scale
        for p in self.params:
            pid = id(p)
            if hasattr(p, "W"):
                for attr, mom in [(("W", self.m)), (("b", self.mb))]:
                    if mom[pid] is None: continue
                    g, w = getattr(p, "d" + attr[0]), getattr(p, attr[0])
                    u = np.sign(self.b1 * mom[pid] + (1.0 - self.b1) * g)
                    w -= lr * (u + self.wd * w if attr[0] == "W" else u)
                    mom[pid] = self.b2 * mom[pid] + (1.0 - self.b2) * g
                    setattr(p, attr[0], w)
            else:
                u = np.sign(self.b1 * self.m[pid] + (1.0 - self.b1) * p.dg)
                p.g -= lr * (u + self.wd * p.g)
                self.m[pid] = self.b2 * self.m[pid] + (1.0 - self.b2) * p.dg

def main():
    N, D, C = 4096, 784, 10
    X = (np.random.randn(N, D) * 0.1).astype("f")
    Y = np.random.randint(0, C, N)
    model = OMEGA_ASI(i=D, h=128, o=C, depth=4)
    params = model.get_params()
    opt = Lion(params, lr=5e-4, wd=0.05)
    bs, epochs = 128, 50

    for epoch in range(epochs):
        idx = np.random.permutation(N)
        l_sum, a_sum = 0, 0
        lr_s = (epoch + 1) / 10 if epoch < 10 else 0.5 * (1 + np.cos(np.pi * (epoch - 10) / (epochs - 10)))

        for i in range(0, N, bs):
            xb, yb = X[idx[i:i+bs]], Y[idx[i:i+bs]]
            logits = model.forward(xb)
            probs = FastOps.softmax(logits)
            l_sum += -np.log(probs[range(len(yb)), yb] + 1e-10).sum()
            a_sum += (probs.argmax(1) == yb).sum()

            dout = probs.copy()
            dout[range(len(yb)), yb] -= 1
            model.backward(dout / len(yb))

            gn = np.sqrt(sum((getattr(p, "dW", 0)**2).sum() + (getattr(p, "db", 0)**2).sum() + (getattr(p, "dg", 0)**2).sum() for p in params))
            if gn > 1.0:
                for p in params:
                    if hasattr(p, "dW"): p.dW /= gn; p.db /= gn
                    if hasattr(p, "dg"): p.dg /= gn

            opt.step(lr_s)

        print(f"EPOCH {epoch+1:02d} | LOSS: {l_sum/N:.4f} | ACC: {a_sum/N:.4f} | LR: {opt.lr*lr_s:.6f}")

if __name__ == "__main__":
    main()
