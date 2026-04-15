import numpy as np

class L: # Linear
    def __init__(self, i, o):
        self.W = (np.random.randn(i, o) * (2/i)**.5).astype('f4')
        self.b = np.zeros(o, 'f4')
    def forward(self, x):
        self.x = x
        return x @ self.W + self.b
    def backward(self, dy):
        self.dW = self.x.reshape(-1, self.x.shape[-1]).T @ dy.reshape(-1, dy.shape[-1])
        self.db = dy.sum(axis=tuple(range(dy.ndim - 1)))
        return dy @ self.W.T

class RN: # RMSNorm
    def __init__(self, d, e=1e-6):
        self.g, self.e = np.ones(d, 'f4'), e
    def forward(self, x):
        self.x = x
        self.r = np.sqrt(np.mean(x**2, -1, keepdims=True) + self.e)
        return self.g * (x / self.r)
    def backward(self, dy):
        xn = self.x / self.r
        self.dg = (dy * xn).sum(axis=tuple(range(dy.ndim-1)))
        dn = dy * self.g
        return (dn - xn * np.mean(dn * xn, -1, keepdims=True)) / self.r

class RoPE:
    def __init__(self, d, m=2048):
        f = 1.0 / (10000 ** (np.arange(0, d, 2)[:d//2] / d))
        t = np.outer(np.arange(m), f)
        self.cos, self.sin = np.cos(t), np.sin(t)
    def apply(self, x, conj=False):
        b, s, h, d = x.shape
        x1, x2 = x[..., :d//2], x[..., d//2:]
        c, sn = self.cos[:s, None, :], self.sin[:s, None, :]
        return np.concatenate([x1*c+x2*sn, x2*c-x1*sn],-1) if conj else np.concatenate([x1*c-x2*sn, x2*c+x1*sn],-1)

class GQA:
    def __init__(self, d, h=8, g=2):
        self.d, self.h, self.g, self.hd = d, h, g, d // h
        self.wq, self.wk, self.wv, self.wo = L(d,d), L(d,d//g), L(d,d//g), L(d,d)
        self.rope, self.sc = RoPE(self.hd), (d//h)**-0.5
    def forward(self, x):
        b, s, _ = x.shape
        q = self.wq.forward(x).reshape(b, s, self.h, self.hd)
        k = self.wk.forward(x).reshape(b, s, self.h//self.g, self.hd)
        v = self.wv.forward(x).reshape(b, s, self.h//self.g, self.hd)
        self.qr, self.kr, self.v_v = self.rope.apply(q), self.rope.apply(k), v
        ke, ve = np.repeat(self.kr, self.g, 2), np.repeat(v, self.g, 2)
        at = np.einsum("bshd,bthd->bsht", self.qr, ke) * self.sc
        e = np.exp(at - at.max(-1, keepdims=True))
        self.p = e / (e.sum(-1, keepdims=True) + 1e-12)
        return self.wo.forward(np.einsum("bsht,bthd->bshd", self.p, ve).reshape(b, s, -1))
    def backward(self, dy):
        b, s, _ = dy.shape
        dy_wo = self.wo.backward(dy).reshape(b, s, self.h, self.hd)
        ke, ve = np.repeat(self.kr, self.g, 2), np.repeat(self.v_v, self.g, 2)
        dp = np.einsum("bshd,bthd->bsht", dy_wo, ve)
        da = self.p * (dp - (self.p * dp).sum(-1, keepdims=True)) * self.sc
        dq = self.rope.apply(np.einsum("bsht,bthd->bshd", da, ke), 1)
        dk_e = np.einsum("bsht,bshd->bthd", da, self.qr)
        dv_e = np.einsum("bsht,bshd->bthd", self.p, dy_wo)
        dk = self.rope.apply(dk_e.reshape(b, s, self.h//self.g, self.g, self.hd).sum(3), 1)
        dv = dv_e.reshape(b, s, self.h//self.g, self.g, self.hd).sum(3)
        return self.wq.backward(dq.reshape(b, s, -1)) + self.wk.backward(dk.reshape(b, s, -1)) + self.wv.backward(dv.reshape(b, s, -1))

class DP: # DualPath
    def __init__(self, d, e=4):
        self.w1, self.v1, self.w2 = L(d, d*e), L(d, d*e), L(d*e, d)
        self.wg, self.gt = L(d, d), L(d, 2)
    def swi(self, x, w, v):
        g = x @ w
        s = g * (1 / (1 + np.exp(-np.clip(g, -12, 12))))
        return s * (x @ v), s
    def forward(self, x):
        self.x = x
        self.g_o, self.s = self.swi(x, self.w1.W, self.v1.W)
        self.g_o = self.g_o @ self.w2.W
        self.q_o = self.wg.forward(x)
        logits = self.gt.forward(x)
        exp = np.exp(logits - logits.max(-1, keepdims=True))
        self.rt = exp / exp.sum(-1, keepdims=True)
        return self.rt[..., :1] * self.g_o + self.rt[..., 1:] * self.q_o
    def backward(self, dy):
        r0, r1 = self.rt[..., :1], self.rt[..., 1:]
        dr = np.stack([(dy * self.g_o).sum(-1), (dy * self.q_o).sum(-1)], -1)
        dg = self.gt.backward(self.rt * (dr - (self.rt * dr).sum(-1, keepdims=True)))
        dx_q = self.wg.backward(dy * r1)
        dy_g = dy * r0
        self.w2.dW = self.s.reshape(-1, self.s.shape[-1]).T @ dy_g.reshape(-1, dy_g.shape[-1])
        ds = dy_g @ self.w2.W.T
        g = self.x @ self.w1.W
        sig = 1 / (1 + np.exp(-np.clip(g, -12, 12)))
        dmg = ds * (self.x @ self.v1.W) * (sig + (g * sig) * (1 - sig))
        dv = ds * (g * sig)
        dx_g = dmg @ self.w1.W.T + dv @ self.v1.W.T
        self.w1.dW, self.v1.dW = self.x.reshape(-1, self.x.shape[-1]).T @ dmg.reshape(-1, dmg.shape[-1]), self.x.reshape(-1, self.x.shape[-1]).T @ dv.reshape(-1, dv.shape[-1])
        return dx_q + dx_g + dg

class Block:
    def __init__(self, d):
        self.n1, self.at, self.n2, self.dp = RN(d), GQA(d), RN(d), DP(d)
    def forward(self, x):
        x = x + self.at.forward(self.n1.forward(x))
        return x + self.dp.forward(self.n2.forward(x))
    def backward(self, dy):
        dy_p = self.dp.backward(dy)
        dy = dy + self.n2.backward(dy_p)
        dy_a = self.at.backward(dy)
        return dy + self.n1.backward(dy_a)

class OMEGA:
    def __init__(self, i, h, o, d=3):
        self.emb = L(i, h)
        self.blks = [Block(h) for _ in range(d)]
        self.ln = RN(h)
        self.head = L(h, o)
    def forward(self, x):
        x = self.emb.forward(x[:, None, :] if x.ndim == 2 else x)
        for b in self.blks: x = b.forward(x)
        return self.head.forward(self.ln.forward(x[:, -1]))
    def backward(self, dy):
        dy = self.ln.backward(self.head.backward(dy))
        dys = np.zeros((dy.shape[0], 1, dy.shape[1]), 'f4')
        dys[:, -1] = dy
        for b in reversed(self.blks): dys = b.backward(dys)
        self.emb.backward(dys)
    def params(self):
        p = []
        def f(o):
            if isinstance(o, (L, RN)): p.append(o)
            elif isinstance(o, list): [f(i) for i in o]
            elif hasattr(o, "__dict__"): [f(v) for k, v in o.__dict__.items() if k not in ('x', 'r', 'p', 'rt', 'g_o', 'q_o', 'qr', 'kr', 'v_v', 's')]
        f(self); return list(set(p))

class Lion:
    def __init__(self, p, lr=1e-4, b1=0.9, b2=0.99, wd=0.01):
        self.p, self.lr, self.b1, self.b2, self.wd = p, lr, b1, b2, wd
        self.m = {id(x): [np.zeros_like(getattr(x, a)) for a in (['W', 'b'] if hasattr(x, 'W') else ['g'])] for x in p}
    def step(self, s=1.0):
        for x in self.p:
            attrs = ['W', 'b'] if hasattr(x, 'W') else ['g']
            for i, a in enumerate(attrs):
                g = getattr(x, 'd'+a if a!='g' else 'dg')
                m = self.m[id(x)][i]
                u = np.sign(self.b1 * m + (1 - self.b1) * g)
                v = getattr(x, a)
                v -= self.lr * s * (u + self.wd * v if a in ('W', 'g') else u)
                self.m[id(x)][i] = self.b2 * m + (1 - self.b2) * g
                setattr(x, a, v)

def train():
    N, D, C, BS, EP = 1024, 784, 10, 64, 20
    X, Y = (np.random.randn(N, D)*0.01).astype('f4'), np.random.randint(0, C, N)
    m = OMEGA(D, 64, C)
    ps = m.params()
    opt = Lion(ps, 2e-4)
    for e in range(EP):
        idx = np.random.permutation(N)
        l_sum, a_sum = 0, 0
        sc = 0.5 * (1 + np.cos(np.pi * e / EP))
        for i in range(0, N, BS):
            xb, yb = X[idx[i:i+BS]], Y[idx[i:i+BS]]
            lg = m.forward(xb)
            pr = np.exp(lg - lg.max(-1, 1))
            pr /= pr.sum(-1, 1)
            l_sum += -np.log(pr[range(len(yb)), yb] + 1e-12).sum()
            a_sum += (pr.argmax(1) == yb).sum()
            dy = pr.copy(); dy[range(len(yb)), yb] -= 1
            m.backward(dy / len(yb))
            gn = np.sqrt(sum((getattr(p, 'd'+a if a!='g' else 'dg')**2).sum() for p in ps for a in (['W', 'b'] if hasattr(p, 'W') else ['g'])))
            if gn > 1.0:
                for p in ps:
                    for a in (['W', 'b'] if hasattr(p, 'W') else ['g']):
                        k = 'd'+a if a!='g' else 'dg'
                        setattr(p, k, getattr(p, k)/gn)
            opt.step(sc)
        print(f"Ep {e} | Loss: {l_sum/N:.3f} | Acc: {a_sum/N:.3f}")

if __name__ == "__main__": train()