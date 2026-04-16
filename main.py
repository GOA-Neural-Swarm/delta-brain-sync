import numpy as np

def silu(x): return x / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))
def dsilu(x):
    s = 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))
    return s * (1.0 + x * (1.0 - s))

class Linear:
    def __init__(self, i, o):
        self.W = np.random.randn(i, o).astype("f4") * np.sqrt(2.0 / i)
        self.b = np.zeros(o, "f4")
    def forward(self, x):
        self.x = x
        return x @ self.W + self.b
    def backward(self, dy):
        self.dW = self.x.reshape(-1, self.x.shape[-1]).T @ dy.reshape(-1, dy.shape[-1])
        self.db = dy.reshape(-1, dy.shape[-1]).sum(axis=0)
        return (dy @ self.W.T).reshape(self.x.shape)

class RMSNorm:
    def __init__(self, d):
        self.g, self.eps = np.ones(d, "f4"), 1e-6
    def forward(self, x):
        self.x = x
        self.rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        self.xn = x / self.rms
        return self.g * self.xn
    def backward(self, dy):
        dn = dy * self.g
        self.dg = np.sum(dy * self.xn, axis=tuple(range(dy.ndim - 1)))
        return (dn - self.xn * np.mean(dn * self.xn, axis=-1, keepdims=True)) / self.rms

class RoPE:
    def __init__(self, d, m=4096):
        f = 10000.0**-(np.arange(0, d, 2) / d)
        t = np.outer(np.arange(m), f)
        self.cos, self.sin = np.cos(t), np.sin(t)
    def apply(self, x, conj=False):
        s, d = x.shape[1], x.shape[-1]
        x1, x2 = x[..., :d//2], x[..., d//2:]
        c, sn = self.cos[:s, None, :], self.sin[:s, None, :]
        return np.concatenate([x1*c+x2*sn, x2*c-x1*sn] if conj else [x1*c-x2*sn, x2*c+x1*sn], -1)

class GQA:
    def __init__(self, d, h=8, g=2):
        self.d, self.h, self.g, self.hd = d, h, g, d // h
        self.wq, self.wk, self.wv, self.wo = Linear(d,d), Linear(d,d//g), Linear(d,d//g), Linear(d,d)
        self.rope, self.scale = RoPE(self.hd), (d//h)**-0.5
    def forward(self, x):
        b, s, _ = x.shape
        q = self.wq.forward(x).reshape(b, s, self.h, self.hd)
        k = self.wk.forward(x).reshape(b, s, self.h//self.g, self.hd)
        v = self.wv.forward(x).reshape(b, s, self.h//self.g, self.hd)
        self.qr, self.kr, self.v_raw = self.rope.apply(q), self.rope.apply(k), v
        ke, ve = np.repeat(self.kr, self.g, 2), np.repeat(v, self.g, 2)
        attn = np.einsum("bshd,bthd->bsht", self.qr, ke) * self.scale
        attn -= np.max(attn, -1, keepdims=True)
        self.p = np.exp(attn); self.p /= (np.sum(self.p, -1, keepdims=True) + 1e-12)
        return self.wo.forward(np.einsum("bsht,bthd->bshd", self.p, ve).reshape(b, s, -1))
    def backward(self, dy):
        b, s, _ = dy.shape
        dy_o = self.wo.backward(dy).reshape(b, s, self.h, self.hd)
        ke, ve = np.repeat(self.kr, self.g, 2), np.repeat(self.v_raw, self.g, 2)
        dp = np.einsum("bshd,bthd->bsht", dy_o, ve)
        da = self.p * (dp - np.sum(self.p * dp, -1, keepdims=True)) * self.scale
        dqr = np.einsum("bsht,bthd->bshd", da, ke)
        dke = np.einsum("bsht,bshd->bthd", da, self.qr)
        dve = np.einsum("bsht,bshd->bthd", self.p, dy_o)
        dq = self.rope.apply(dqr, True).reshape(b, s, -1)
        dk = self.rope.apply(dke.reshape(b, s, self.h//self.g, self.g, self.hd).sum(3), True).reshape(b, s, -1)
        dv = dve.reshape(b, s, self.h//self.g, self.g, self.hd).sum(3).reshape(b, s, -1)
        return self.wq.backward(dq) + self.wk.backward(dk) + self.wv.backward(dv)

class MoE:
    def __init__(self, d, n=4):
        self.n, self.gate = n, Linear(d, n)
        self.experts = [Linear(d, d*2) for _ in range(n)]
        self.out_proj = [Linear(d*2, d) for _ in range(n)]
    def forward(self, x):
        self.x = x
        g = self.gate.forward(x)
        self.p = np.exp(g - np.max(g, -1, keepdims=True))
        self.p /= np.sum(self.p, -1, keepdims=True)
        self.ex_out = [op.forward(silu(ex.forward(x))) for ex, op in zip(self.experts, self.out_proj)]
        return sum(self.p[..., i:i+1] * self.ex_out[i] for i in range(self.n))
    def backward(self, dy):
        dx, dp = np.zeros_like(self.x), np.zeros_like(self.p)
        for i in range(self.n):
            pi = self.p[..., i:i+1]
            dp[..., i] = np.sum(dy * self.ex_out[i], -1)
            de_out = dy * pi
            ds = self.out_proj[i].backward(de_out)
            dx += self.experts[i].backward(ds * dsilu(self.experts[i].x))
        dg = self.p * (dp - np.sum(self.p * dp, -1, keepdims=True))
        return dx + self.gate.backward(dg)

class Block:
    def __init__(self, d):
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn, self.moe = GQA(d), MoE(d)
    def forward(self, x):
        x = x + self.attn.forward(self.n1.forward(x))
        return x + self.moe.forward(self.n2.forward(x))
    def backward(self, dy):
        dy = dy + self.n2.backward(self.moe.backward(dy))
        return dy + self.n1.backward(self.attn.backward(dy))

class SovereignASI:
    def __init__(self, i, h, o, depth=2):
        self.emb, self.blocks = Linear(i, h), [Block(h) for _ in range(depth)]
        self.norm, self.head = RMSNorm(h), Linear(h, o)
    def forward(self, x):
        x = self.emb.forward(x[:, None, :] if x.ndim == 2 else x)
        for b in self.blocks: x = b.forward(x)
        self.last_x = self.norm.forward(x[:, -1, :])
        return self.head.forward(self.last_x)
    def backward(self, dy):
        dy = self.norm.backward(self.head.backward(dy))
        dy_s = np.zeros((dy.shape[0], self.emb.x.shape[1], dy.shape[1]), "f4")
        dy_s[:, -1, :] = dy
        for b in reversed(self.blocks): dy_s = b.backward(dy_s)
        self.emb.backward(dy_s)

class AdamW:
    def __init__(self, model, lr=1e-3, wd=0.01):
        self.p = self._get_p(model); self.lr, self.wd, self.t = lr, wd, 0
        self.m = [np.zeros_like(x) for x in self.p]
        self.v = [np.zeros_like(x) for x in self.p]
    def _get_p(self, obj):
        ps = []
        if isinstance(obj, Linear): ps += [obj.W, obj.b]
        elif isinstance(obj, RMSNorm): ps += [obj.g]
        elif isinstance(obj, list): [ps.extend(self._get_p(i)) for i in obj]
        elif hasattr(obj, "__dict__"): [ps.extend(self._get_p(v)) for k, v in obj.__dict__.items() if k != "rope"]
        return ps
    def _get_g(self, obj):
        gs = []
        if isinstance(obj, Linear): gs += [obj.dW, obj.db]
        elif isinstance(obj, RMSNorm): gs += [obj.dg]
        elif isinstance(obj, list): [gs.extend(self._get_g(i)) for i in obj]
        elif hasattr(obj, "__dict__"): [gs.extend(self._get_g(v)) for k, v in obj.__dict__.items() if k != "rope"]
        return gs
    def step(self, model):
        self.t += 1; gs = self._get_g(model)
        for i, (p, g) in enumerate(zip(self.p, gs)):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * (g**2)
            mt, vt = self.m[i] / (1-0.9**self.t), self.v[i] / (1-0.999**self.t)
            p -= self.lr * (mt / (np.sqrt(vt) + 1e-8) + self.wd * p)

def train():
    N, D, C, BS, E = 128, 784, 10, 16, 20
    X, Y = np.random.randn(N, D).astype("f4"), np.random.randint(0, C, N)
    m = SovereignASI(D, 64, C, 1); opt = AdamW(m, 1e-3)
    for e in range(E):
        idx = np.random.permutation(N); tl, ta = 0, 0
        for i in range(0, N, BS):
            xb, yb = X[idx[i:i+BS]], Y[idx[i:i+BS]]
            lg = m.forward(xb)
            pr = np.exp(lg - np.max(lg, -1, keepdims=True)); pr /= np.sum(pr, -1, keepdims=True)
            tl += -np.mean(np.log(pr[np.arange(len(yb)), yb] + 1e-12)) * len(yb)
            ta += np.sum(np.argmax(pr, -1) == yb)
            dy = pr.copy(); dy[np.arange(len(yb)), yb] -= 1
            m.backward(dy / len(yb)); opt.step(m)
        if (e+1) % 5 == 0: print(f"E {e+1} | Loss: {tl/N:.4f} | Acc: {ta/N:.4f}")

if __name__ == "__main__": train()