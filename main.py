import numpy as np

class L:
    def __init__(self, i, o, s=0.02):
        self.W = np.random.randn(i, o).astype('f4') * (s if s else (2/i)**.5)
        self.b = np.zeros(o, 'f4')
    def f(self, x):
        self.x = x
        return x @ self.W + self.b
    def bwd(self, dy):
        self.dW = self.x.reshape(-1, self.x.shape[-1]).T @ dy.reshape(-1, dy.shape[-1])
        self.db = dy.sum(axis=tuple(range(dy.ndim-1)))
        return dy @ self.W.T

class N:
    def __init__(self, d, e=1e-6):
        self.g, self.e = np.ones(d, 'f4'), e
    def f(self, x):
        self.x = x
        self.rms = np.sqrt(np.mean(x**2, -1, keepdims=True) + self.e)
        return self.g * (x / self.rms)
    def bwd(self, dy):
        xn = self.x / self.rms
        self.dg = (dy * xn).sum(axis=tuple(range(dy.ndim-1)))
        dn = dy * self.g
        return (dn - xn * np.mean(dn * xn, -1, keepdims=True)) / self.rms

class R:
    def __init__(self, d, m=2048):
        f = 1.0 / (10000**(np.arange(0, d, 2) / d))
        t = np.outer(np.arange(m), f)
        self.c, self.s = np.cos(t), np.sin(t)
    def a(self, x, conj=False):
        b, s, h, d = x.shape
        x1, x2 = x[..., :d//2], x[..., d//2:]
        c, sn = self.c[:s, None, :], self.s[:s, None, :]
        if conj: return np.concatenate([x1*c+x2*sn, x2*c-x1*sn], -1)
        return np.concatenate([x1*c-x2*sn, x2*c+x1*sn], -1)

class GQA:
    def __init__(self, d, h=8, g=2):
        self.d, self.h, self.g, self.hd = d, h, g, d//h
        self.wq, self.wk, self.wv, self.wo = L(d, d), L(d, d//g), L(d, d//g), L(d, d)
        self.rope, self.sc = R(self.hd), (d//h)**-0.5
    def f(self, x):
        b, s, _ = x.shape
        q = self.wq.f(x).reshape(b, s, self.h, self.hd)
        k = self.wk.f(x).reshape(b, s, self.h//self.g, self.hd)
        v = self.wv.f(x).reshape(b, s, self.h//self.g, self.hd)
        self.qr, self.kr, self.v_v = self.rope.a(q), self.rope.a(k), v
        ke, ve = np.repeat(self.kr, self.g, 2), np.repeat(v, self.g, 2)
        at = np.einsum("bshd,bthd->bsht", self.qr, ke) * self.sc
        e = np.exp(at - at.max(-1, keepdims=True))
        self.p = e / (e.sum(-1, keepdims=True) + 1e-12)
        return self.wo.f(np.einsum("bsht,bthd->bshd", self.p, ve).reshape(b, s, -1))
    def bwd(self, dy):
        b, s, _ = dy.shape
        dy_wo = self.wo.bwd(dy).reshape(b, s, self.h, self.hd)
        ke, ve = np.repeat(self.kr, self.g, 2), np.repeat(self.v_v, self.g, 2)
        dp = np.einsum("bshd,bthd->bsht", dy_wo, ve)
        da = self.p * (dp - (self.p * dp).sum(-1, keepdims=True)) * self.sc
        dq = self.rope.a(np.einsum("bsht,bthd->bshd", da, ke), 1)
        dk_e = np.einsum("bsht,bshd->bthd", da, self.qr)
        dv_e = np.einsum("bsht,bshd->bthd", self.p, dy_wo)
        dk = self.rope.a(dk_e.reshape(b, s, self.h//self.g, self.g, self.hd).sum(3), 1)
        dv = dv_e.reshape(b, s, self.h//self.g, self.g, self.hd).sum(3)
        return self.wq.bwd(dq.reshape(b, s, -1)) + self.wk.bwd(dk.reshape(b, s, -1)) + self.wv.bwd(dv.reshape(b, s, -1))

class MoE:
    def __init__(self, d, n=4, e=2):
        self.n, self.d = n, d
        self.gate = L(d, n)
        self.experts = [[L(d, d*e), L(d*e, d), L(d, d*e)] for _ in range(n)]
    def f(self, x):
        self.x = x
        g_lg = self.gate.f(x)
        ex_g = np.exp(g_lg - g_lg.max(-1, keepdims=True))
        self.pr = ex_g / ex_g.sum(-1, keepdims=True)
        self.eo = []
        out = np.zeros_like(x)
        for i in range(self.n):
            w1, w2, w3 = self.experts[i]
            x1, x3 = w1.f(x), w3.f(x)
            sig = 1 / (1 + np.exp(-np.clip(x1, -12, 12)))
            sw = x1 * sig
            o = w2.f(sw * x3)
            self.eo.append((x1, x3, sig, sw, o))
            out += self.pr[..., i:i+1] * o
        return out
    def bwd(self, dy):
        dx = np.zeros_like(self.x)
        dp = np.zeros_like(self.pr)
        for i in range(self.n):
            x1, x3, sig, sw, o = self.eo[i]
            w1, w2, w3 = self.experts[i]
            dp[..., i] = (dy * o).sum(-1)
            dy_exp = dy * self.pr[..., i:i+1]
            dw2 = w2.bwd(dy_exp)
            dx3, dsw = dw2 * sw, dw2 * x3
            dx1 = dsw * (sig + x1 * sig * (1 - sig))
            dx += w1.bwd(dx1) + w3.bwd(dx3)
        dl = self.pr * (dp - (self.pr * dp).sum(-1, keepdims=True))
        return dx + self.gate.bwd(dl)

class RedundantConsensus:
    def __init__(self, d):
        self.path_a = GQA(d)
        self.path_b = MoE(d)
        self.w = np.array([0.5, 0.5], 'f4')
    def f(self, x):
        self.oa, self.ob = self.path_a.f(x), self.path_b.f(x)
        return self.w[0] * self.oa + self.w[1] * self.ob
    def bwd(self, dy):
        da = self.path_a.bwd(dy * self.w[0])
        db = self.path_b.bwd(dy * self.w[1])
        self.dw = np.array([(dy * self.oa).sum(), (dy * self.ob).sum()], 'f4')
        return da + db

class Block:
    def __init__(self, d):
        self.n1, self.rc, self.n2, self.moe = N(d), RedundantConsensus(d), N(d), MoE(d)
    def f(self, x):
        x = x + self.rc.f(self.n1.f(x))
        return x + self.moe.f(self.n2.f(x))
    def bwd(self, dy):
        dy = dy + self.n2.bwd(self.moe.bwd(dy))
        return dy + self.n1.bwd(self.rc.bwd(dy))

class OMEGA_ASI:
    def __init__(self, i, h, o, d=2):
        self.emb = L(i, h)
        self.blks = [Block(h) for _ in range(d)]
        self.ln = N(h)
        self.head = L(h, o)
    def f(self, x):
        x = self.emb.f(x[:, None] if x.ndim == 2 else x)
        for b in self.blks: x = b.f(x)
        return self.head.f(self.ln.f(x[:, -1]))
    def bwd(self, dy):
        dy = self.ln.bwd(self.head.bwd(dy))
        dys = np.zeros((dy.shape[0], 1, dy.shape[1]), 'f4')
        dys[:, -1] = dy
        for b in reversed(self.blks): dys = b.bwd(dys)
        self.emb.bwd(dys)
    def params(self):
        p = []
        def g(obj):
            if isinstance(obj, (L, N)): p.append(obj)
            elif isinstance(obj, list): [g(i) for i in obj]
            elif hasattr(obj, "__dict__"): [g(v) for k, v in obj.__dict__.items() if k[0]!='_']
        g(self); return list(set(p))

class AdamW:
    def __init__(self, p, lr=2e-3, b1=.9, b2=.95, wd=.01):
        self.p, self.lr, self.b1, self.b2, self.wd, self.t = p, lr, b1, b2, wd, 0
        self.m = {id(x): [np.zeros_like(getattr(x, a)) for a in (['W','b'] if hasattr(x,'W') else ['g'])] for x in p}
        self.v = {id(x): [np.zeros_like(getattr(x, a)) for a in (['W','b'] if hasattr(x,'W') else ['g'])] for x in p}
    def step(self):
        self.t += 1
        lr_t = self.lr * min(1.0, self.t / 100) * (0.5 * (1 + np.cos(self.t * np.pi / 1000)))
        for x in self.p:
            at = ['W', 'b'] if hasattr(x, 'W') else ['g']
            for i, a in enumerate(at):
                gr = getattr(x, 'd'+a if a!='g' else 'dg')
                m, v = self.m[id(x)][i], self.v[id(x)][i]
                m[:] = self.b1 * m + (1-self.b1) * gr
                v[:] = self.b2 * v + (1-self.b2) * (gr**2)
                mh, vh = m/(1-self.b1**self.t), v/(1-self.b2**self.t)
                pv = getattr(x, a)
                pv -= lr_t * (mh/(np.sqrt(vh)+1e-8) + self.wd * pv)
                setattr(x, a, pv)

def train():
    N_SAMP, DIM, CATS, BS, EPS = 1024, 784, 10, 32, 50
    X = (np.random.randn(N_SAMP, DIM) * 0.01).astype('f4')
    Y = np.random.randint(0, CATS, N_SAMP)
    model = OMEGA_ASI(DIM, 128, CATS, d=2)
    ps = model.params()
    opt = AdamW(ps, lr=1e-3)
    
    for e in range(EPS):
        idx = np.random.permutation(N_SAMP)
        l_sum, a_sum = 0, 0
        for i in range(0, N_SAMP, BS):
            xb, yb = X[idx[i:i+BS]], Y[idx[i:i+BS]]
            logits = model.f(xb)
            probs = np.exp(logits - logits.max(-1, keepdims=True))
            probs /= probs.sum(-1, keepdims=True)
            l_sum += -np.log(probs[range(len(yb)), yb] + 1e-12).sum()
            a_sum += (probs.argmax(1) == yb).sum()
            dy = probs.copy()
            dy[range(len(yb)), yb] -= 1
            model.bwd(dy / len(yb))
            
            gn = np.sqrt(sum((getattr(p,'d'+a if a!='g' else 'dg')**2).sum() for p in ps for a in (['W','b'] if hasattr(p,'W') else ['g'])))
            if gn > 1.0:
                for p in ps:
                    for a in (['W','b'] if hasattr(p,'W') else ['g']):
                        k = 'd'+a if a!='g' else 'dg'
                        setattr(p, k, getattr(p, k) / gn)
            opt.step()
        print(f"STEP {e+1:03d} | LOSS: {l_sum/N_SAMP:.4f} | ACC: {a_sum/N_SAMP:.4f}")

if __name__ == "__main__":
    train()
