import numpy as np

class Tensor:
    def __init__(self, data, name=""):
        self.data = data.astype("f4")
        self.grad = np.zeros_like(data)

class Linear:
    def __init__(self, i, o, name=""):
        s = np.sqrt(2. / (i + o))
        self.w = Tensor(np.random.normal(0, s, (i, o)), f"{name}_w")
        self.b = Tensor(np.zeros(o), f"{name}_b")

    def forward(self, x):
        self.x = x
        return x @ self.w.data + self.b.data

    def backward(self, dy):
        self.w.grad += self.x.reshape(-1, self.x.shape[-1]).T @ dy.reshape(-1, dy.shape[-1])
        self.b.grad += dy.reshape(-1, dy.shape[-1]).sum(0)
        return dy @ self.w.data.T

class RMSNorm:
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

class GQA:
    def __init__(self, d, h=8, g=2):
        self.d, self.h, self.g, self.hd = d, h, g, d // h
        self.wq, self.wk, self.wv, self.wo = [Linear(d, d if i!=1 else (h//g)*(d//h)) for i in range(4)]
        self.scale = self.hd**-0.5

    def _rope(self, t, inv=False):
        s = t.shape[1]
        p = np.arange(s)[:, None]
        f = 10000**-(np.arange(0, self.hd, 2)/self.hd)
        a = p * f
        cos, sin = np.cos(a), np.sin(a)
        if inv: sin = -sin
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
        self.q_r, self.k_r, self.v_r = self._rope(q), self._rope(k), v
        kp, vp = np.repeat(self.q_r, 1, 2), np.repeat(self.k_r, self.g, 2)
        v_rep = np.repeat(v, self.g, 2)
        attn = np.einsum("bshd,bthd->bsht", self.q_r, vp) * self.scale
        self.p = np.exp(attn - np.max(attn, -1, keepdims=True))
        self.p /= (self.p.sum(-1, keepdims=True) + 1e-12)
        out = np.einsum("bsht,bthd->bshd", self.p, v_rep).reshape(b, s, -1)
        return self.wo.forward(out)

    def backward(self, dy):
        b, s = dy.shape[:2]
        dy_wo = self.wo.backward(dy).reshape(b, s, self.h, self.hd)
        vp = np.repeat(self.v_r, self.g, 2)
        kp = np.repeat(self.k_r, self.g, 2)
        dp = np.einsum("bshd,bthd->bsht", dy_wo, vp)
        da = self.p * (dp - np.sum(self.p * dp, -1, keepdims=True)) * self.scale
        dq_r = np.einsum("bsht,bthd->bshd", da, kp)
        dk_r = np.einsum("bsht,bshd->bthd", da, self.q_r).reshape(b, s, self.h//self.g, self.g, self.hd).sum(3)
        dv = np.einsum("bsht,bshd->bthd", self.p, dy_wo).reshape(b, s, self.h//self.g, self.g, self.hd).sum(3)
        dq = self.wq.backward(self._rope(dq_r, True).reshape(b, s, -1))
        dk = self.wk.backward(self._rope(dk_r, True).reshape(b, s, -1))
        dv = self.wv.backward(dv.reshape(b, s, -1))
        return dq + dk + dv

class MoE:
    def __init__(self, d, n=4, k=2):
        self.d, self.n, self.k = d, n, k
        self.gate = Linear(d, n)
        self.w1 = [Linear(d, d*2) for _ in range(n)]
        self.w2 = [Linear(d*2, d) for _ in range(n)]

    def _swiglu(self, x, deriv=False, cache=None):
        x, g = np.split(x, 2, -1)
        sig = 1. / (1. + np.exp(-np.clip(g, -15, 15)))
        if deriv:
            dy, (old_x, old_g, old_sig) = cache
            dx = dy * (old_g * old_sig)
            dg = dy * old_x * old_sig * (1. + old_g * (1. - old_sig))
            return np.concatenate([dx, dg], -1)
        return x * (g * sig), (x, g, sig)

    def forward(self, x):
        self.sh = x.shape
        x = x.reshape(-1, self.d)
        logits = self.gate.forward(x)
        p = np.exp(logits - np.max(logits, -1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        self.idx = np.argsort(p, -1)[:, -self.k:]
        self.w = np.take_along_axis(p, self.idx, -1)
        self.w /= (self.w.sum(-1, keepdims=True) + 1e-12)
        out = np.zeros_like(x)
        self.ex_in = []
        for i in range(self.n):
            m = np.any(self.idx == i, -1)
            if not np.any(m): 
                self.ex_in.append(None)
                continue
            pos = np.where(self.idx[m] == i)[1]
            wi = self.w[m, pos][:, None]
            h1 = self.w1[i].forward(x[m])
            act, c = self._swiglu(h1)
            h2 = self.w2[i].forward(act)
            out[m] += h2 * wi
            self.ex_in.append((m, pos, act, c))
        return out.reshape(self.sh)

    def backward(self, dy):
        dyf = dy.reshape(-1, self.d)
        dx, dg = np.zeros_like(dyf), np.zeros((dyf.shape[0], self.n))
        for i in range(self.n):
            if self.ex_in[i] is None: continue
            m, pos, act, c = self.ex_in[i]
            wi = self.w[m, pos][:, None]
            dyi = dyf[m] * wi
            dg[m, i] = np.sum(dyf[m] * self.w2[i].forward(act), -1)
            dact = self.w2[i].backward(dyi)
            dh1 = self._swiglu(None, True, (dact, c))
            dx[m] += self.w1[i].backward(dh1)
        return (dx + self.gate.backward(dg - np.mean(dg, -1, keepdims=True))).reshape(self.sh)

class SovereignBlock:
    def __init__(self, d):
        self.n1, self.at, self.n2, self.moe = RMSNorm(d), GQA(d), RMSNorm(d), MoE(d)
        self.pg = Tensor(np.array([0.5]))

    def forward(self, x):
        x = x + self.at.forward(self.n1.forward(x))
        self.moe_res = x
        m_out = self.moe.forward(self.n2.forward(x))
        return x + self.pg.data * m_out + (1 - self.pg.data) * x

    def backward(self, dy):
        dm = dy * self.pg.data
        dx_m = self.moe.backward(self.n2.backward(dm))
        dy = dy + dx_m + dy * (1 - self.pg.data)
        return dy + self.at.backward(self.n1.backward(dy))

class OMEGA_ASI:
    def __init__(self, di, dm, do, depth=2):
        self.embed = Linear(di, dm)
        self.blocks = [SovereignBlock(dm) for _ in range(depth)]
        self.fn, self.head = RMSNorm(dm), Linear(dm, do)
        self.params = self._get_p()

    def _get_p(self):
        p = []
        def _w(o):
            if isinstance(o, Tensor): p.append(o)
            elif hasattr(o, 'w'): p.extend([o.w, o.b])
            elif hasattr(o, 'g'): p.append(o.g)
            elif isinstance(o, list): [_w(i) for i in o]
            elif hasattr(o, '__dict__'): [_w(v) for v in o.__dict__.values()]
        _w(self)
        return p

    def forward(self, x):
        x = self.embed.forward(x[:, None] if x.ndim==2 else x)
        for b in self.blocks: x = b.forward(x)
        return self.head.forward(self.fn.forward(x[:, -1]))

    def backward(self, dy):
        dy = self.fn.backward(self.head.backward(dy))
        db = np.zeros((dy.shape[0], self.embed.x.shape[1], dy.shape[1]))
        db[:, -1] = dy
        for b in reversed(self.blocks): db = b.backward(db)
        self.embed.backward(db)

class AdamW:
    def __init__(self, p, lr=1e-3, wd=0.01):
        self.p, self.lr, self.wd = p, lr, wd
        self.m = [np.zeros_like(i.data) for i in p]
        self.v = [np.zeros_like(i.data) for i in p]
        self.t = 0

    def step(self):
        self.t += 1
        lt = self.lr * (np.sqrt(1-0.999**self.t)/(1-0.9**self.t))
        for i, pt in enumerate(self.p):
            g = np.clip(pt.grad, -1, 1)
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * (g**2)
            pt.data -= lt * (self.m[i] / (np.sqrt(self.v[i]) + 1e-8) + self.wd * pt.data)
            pt.grad.fill(0)

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
            lgt = m.forward(xb)
            pr = np.exp(lgt - np.max(lgt, -1, keepdims=True))
            pr /= pr.sum(-1, keepdims=True)
            ls.append(-np.mean(np.log(pr[np.arange(len(yb)), yb] + 1e-12)))
            ac.append(np.mean(np.argmax(pr, -1) == yb))
            dl = pr.copy()
            dl[np.arange(len(yb)), yb] -= 1
            m.backward(dl / len(yb))
            opt.step()
        if (e + 1) % 5 == 0:
            print(f"E {e+1:03d} | L {np.mean(ls):.4f} | A {np.mean(ac):.4f}")

if __name__ == "__main__":
    train()