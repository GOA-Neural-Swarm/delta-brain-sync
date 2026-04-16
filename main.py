import numpy as np

def silu(x): return x / (1.0 + np.exp(-np.clip(x, -15, 15)))
def dsilu(x):
    s = 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))
    return s * (1.0 + x * (1.0 - s))

class Linear:
    def __init__(self, i, o):
        self.W = np.random.randn(i, o).astype("f4") * np.sqrt(2/i)
        self.b = np.zeros(o, "f4")
    def forward(self, x):
        self.x = x
        return x @ self.W + self.b
    def backward(self, dy):
        self.dW = self.x.reshape(-1, self.x.shape[-1]).T @ dy.reshape(-1, dy.shape[-1])
        self.db = dy.reshape(-1, dy.shape[-1]).sum(0)
        return (dy @ self.W.T).reshape(self.x.shape)

class RMS:
    def __init__(self, d, e=1e-6): self.g, self.e = np.ones(d, "f4"), e
    def forward(self, x):
        self.x = x
        self.r = np.sqrt(np.mean(x**2, -1, keepdims=True) + self.e)
        self.xn = x / self.r
        return self.g * self.xn
    def backward(self, dy):
        dn = dy * self.g
        self.dg = np.sum(dy * self.xn, axis=tuple(range(dy.ndim - 1)))
        return (dn - self.xn * np.mean(dn * self.xn, -1, keepdims=True)) / self.r

class RoPE:
    def __init__(self, d, m=4096):
        f = 10000.0**-(np.arange(0, d, 2) / d)
        t = np.outer(np.arange(m), f)
        self.c, self.s = np.cos(t), np.sin(t)
    def apply(self, x, conj=False):
        s, d = x.shape[1], x.shape[-1]
        x1, x2 = x[..., :d//2], x[..., d//2:]
        c, sn = self.c[:s, None, :], self.s[:s, None, :]
        return np.concatenate([x1*c+x2*sn, x2*c-x1*sn], -1) if conj else np.concatenate([x1*c-x2*sn, x2*c+x1*sn], -1)

class GQA:
    def __init__(self, d, h=8, g=2):
        self.d, self.h, self.g, self.hd = d, h, g, d // h
        self.wq, self.wk, self.wv, self.wo = [Linear(d, d if i==0 or i==3 else d//g) for i in range(4)]
        self.rope, self.sc = RoPE(self.hd), (d//h)**-0.5
    def forward(self, x):
        b, s, _ = x.shape
        q = self.wq.forward(x).reshape(b, s, self.h, self.hd)
        k = self.wk.forward(x).reshape(b, s, self.h//self.g, self.hd)
        v = self.wv.forward(x).reshape(b, s, self.h//self.g, self.hd)
        self.qr, self.kr, self.v_raw = self.rope.apply(q), self.rope.apply(k), v
        ke, ve = np.repeat(self.kr, self.g, 2), np.repeat(v, self.g, 2)
        at = np.einsum("bshd,bthd->bsht", self.qr, ke) * self.sc
        at -= np.max(at, -1, keepdims=True)
        self.p = np.exp(at); self.p /= (np.sum(self.p, -1, keepdims=True) + 1e-12)
        return self.wo.forward(np.einsum("bsht,bthd->bshd", self.p, ve).reshape(b, s, -1))
    def backward(self, dy):
        b, s, _ = dy.shape
        dyo = self.wo.backward(dy).reshape(b, s, self.h, self.hd)
        ke, ve = np.repeat(self.kr, self.g, 2), np.repeat(self.v_raw, self.g, 2)
        dp = np.einsum("bshd,bthd->bsht", dyo, ve)
        da = self.p * (dp - np.sum(self.p * dp, -1, keepdims=True)) * self.sc
        dqr = np.einsum("bsht,bthd->bshd", da, ke)
        dkr = np.einsum("bsht,bshd->bthd", da, self.qr).reshape(b, s, self.h//self.g, self.g, self.hd).sum(3)
        dv = np.einsum("bsht,bshd->bthd", self.p, dyo).reshape(b, s, self.h//self.g, self.g, self.hd).sum(3)
        dq, dk = self.rope.apply(dqr, 1).reshape(b, s, -1), self.rope.apply(dkr, 1).reshape(b, s, -1)
        return self.wq.backward(dq) + self.wk.backward(dk) + self.wv.backward(dv.reshape(b, s, -1))

class MoE:
    def __init__(self, d, n=4, k=2):
        self.d, self.n, self.k = d, n, k
        self.gate = Linear(d, n)
        self.ex = [[Linear(d, d*2), Linear(d*2, d)] for _ in range(n)]
    def forward(self, x):
        self.sh = x.shape
        xf = x.reshape(-1, self.d)
        lg = self.gate.forward(xf)
        p = np.exp(lg - np.max(lg, -1, keepdims=True)); p /= p.sum(-1, keepdims=True)
        tk = np.argsort(p, -1)[:, -self.k:]
        mk = np.zeros_like(p); np.put_along_axis(mk, tk, 1.0, -1)
        self.gw = (p * mk); self.gw /= (self.gw.sum(-1, keepdims=True) + 1e-12)
        out = np.zeros_like(xf); self.ein, self.eint = [], []
        for i in range(self.n):
            idx = np.where(mk[:, i] == 1)[0]
            if not len(idx): self.ein.append(None); self.eint.append(None); continue
            ix = xf[idx]; it = silu(self.ex[i][0].forward(ix))
            out[idx] += self.ex[i][1].forward(it) * self.gw[idx, i:i+1]
            self.ein.append(ix); self.eint.append(it)
        return out.reshape(self.sh)
    def backward(self, dy):
        df = dy.reshape(-1, self.d); dx = np.zeros((df.shape[0], self.d), "f4")
        dg = np.zeros((df.shape[0], self.n), "f4")
        for i in range(self.n):
            idx = np.where(self.gw[:, i] > 0)[0]
            if not len(idx): continue
            eo = self.ex[i][1].forward(self.eint[i])
            dg[idx, i] = np.sum(df[idx] * eo, -1)
            ds = self.ex[i][1].backward(df[idx] * self.gw[idx, i:i+1])
            dx[idx] += self.ex[i][0].backward(ds * dsilu(self.ex[i][0].x))
        d_gt = self.gw * (dg - np.sum(self.gw * dg, -1, keepdims=True))
        return (dx + self.gate.backward(d_gt)).reshape(self.sh)

class SovereignBlock:
    def __init__(self, d):
        self.n1, self.n2, self.at, self.moe = RMS(d), RMS(d), GQA(d), MoE(d)
        self.f, self.fn = Linear(d, d), RMS(d)
    def forward(self, x):
        x = x + self.at.forward(self.n1.forward(x))
        x = x + self.moe.forward(self.n2.forward(x))
        self.ix = x
        return x + silu(self.f.forward(self.fn.forward(x)))
    def backward(self, dy):
        df = self.fn.backward(self.f.backward(dy * dsilu(self.f.x)))
        dy = dy + df
        dy = dy + self.n2.backward(self.moe.backward(dy))
        return dy + self.n1.backward(self.at.backward(dy))

class OMEGA_ASI:
    def __init__(self, i, h, o, depth=2):
        self.emb, self.head, self.norm = Linear(i, h), Linear(h, o), RMS(h)
        self.blks = [SovereignBlock(h) for _ in range(depth)]
    def forward(self, x):
        if x.ndim == 2: x = x[:, None, :]
        x = self.emb.forward(x)
        for b in self.blks: x = b.forward(x)
        self.lx = self.norm.forward(x[:, -1, :])
        return self.head.forward(self.lx)
    def backward(self, dy):
        dy = self.norm.backward(self.head.backward(dy))
        dys = np.zeros((dy.shape[0], self.emb.x.shape[1], dy.shape[1]), "f4")
        dys[:, -1, :] = dy
        for b in reversed(self.blks): dys = b.backward(dys)
        self.emb.backward(dys)

class AdamW:
    def __init__(self, m, lr=1e-3, wd=0.01):
        self.lr, self.wd, self.t, self.p = lr, wd, 0, []
        self._get(m)
        self.mv = [[np.zeros_like(x), np.zeros_like(x)] for x in self.p]
    def _get(self, obj):
        if isinstance(obj, Linear): self.p.extend([obj.W, obj.b])
        elif isinstance(obj, RMS): self.p.append(obj.g)
        elif isinstance(obj, (list, tuple)): [self._get(i) for i in obj]
        elif hasattr(obj, "__dict__"): [self._get(v) for k, v in obj.__dict__.items() if k not in ["rope", "x", "xn", "r", "p", "qr", "kr", "v_raw", "gw", "ein", "eint", "lx", "sh", "ix"]]
    def _grads(self, obj, gs):
        if isinstance(obj, Linear): gs.extend([obj.dW, obj.db])
        elif isinstance(obj, RMS): gs.append(obj.dg)
        elif isinstance(obj, (list, tuple)): [self._grads(i, gs) for i in obj]
        elif hasattr(obj, "__dict__"): [self._grads(v, gs) for k, v in obj.__dict__.items() if k not in ["rope", "x", "xn", "r", "p", "qr", "kr", "v_raw", "gw", "ein", "eint", "lx", "sh", "ix"]]
    def step(self, m):
        self.t += 1; gs = []; self._grads(m, gs)
        lt = self.lr * (np.sqrt(1-0.999**self.t)/(1-0.9**self.t))
        for i, (p, g) in enumerate(zip(self.p, gs)):
            self.mv[i][0] = 0.9 * self.mv[i][0] + 0.1 * g
            self.mv[i][1] = 0.999 * self.mv[i][1] + 0.001 * (g**2)
            p -= lt * (self.mv[i][0] / (np.sqrt(self.mv[i][1]) + 1e-8) + self.wd * p)

def train():
    N, D, C, BS, E = 256, 784, 10, 32, 50
    X, Y = np.random.randn(N, D).astype("f4"), np.random.randint(0, C, N)
    m = OMEGA_ASI(D, 128, C)
    opt = AdamW(m, 2e-3)
    for e in range(E):
        idx = np.random.permutation(N); l, a = 0, 0
        for i in range(0, N, BS):
            b_idx = idx[i:i+BS]; xb, yb = X[b_idx], Y[b_idx]
            lg = m.forward(xb)
            pr = np.exp(lg - np.max(lg, -1, keepdims=True)); pr /= pr.sum(-1, keepdims=True)
            l += -np.mean(np.log(pr[np.arange(len(yb)), yb] + 1e-12)) * len(yb)
            a += np.sum(np.argmax(pr, -1) == yb)
            dy = pr.copy(); dy[np.arange(len(yb)), yb] -= 1
            m.backward(dy / len(yb)); opt.step(m)
        if (e + 1) % 5 == 0: print(f"E{e+1} | Loss: {l/N:.4f} | Acc: {a/N:.4f}")

if __name__ == "__main__": train()