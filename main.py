
import numpy as np

class Linear:
    def __init__(self, i, o, s=None):
        self.W = np.random.randn(i, o).astype('f4') * (s if s else np.sqrt(2/i))
        self.b = np.zeros(o, 'f4')
        self.dW, self.db = None, None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dy):
        self.dW = self.x.reshape(-1, self.x.shape[-1]).T @ dy.reshape(-1, dy.shape[-1])
        self.db = dy.sum(axis=tuple(range(dy.ndim-1)))
        return dy @ self.W.T

class RMSNorm:
    def __init__(self, d, e=1e-6):
        self.g, self.e = np.ones(d, 'f4'), e

    def forward(self, x):
        self.x = x
        self.r = 1 / np.sqrt(np.mean(x**2, -1, keepdims=True) + self.e)
        return self.g * (x * self.r)

    def backward(self, dy):
        xn = self.x * self.r
        self.dg = (dy * xn).sum(axis=tuple(range(dy.ndim-1)))
        dn = dy * self.g
        return self.r * (dn - xn * np.mean(dn * xn, -1, keepdims=True))

class RoPE:
    def __init__(self, d, m=2048):
        f = 1.0 / (10000**(np.arange(0, d, 2) / d))
        t = np.outer(np.arange(m), f)
        self.cos, self.sin = np.cos(t), np.sin(t)

    def apply(self, x, conj=False):
        b, s, h, d = x.shape
        x1, x2 = x[..., :d//2], x[..., d//2:]
        c, sn = self.cos[:s, None, :], self.sin[:s, None, :]
        if conj: return np.concatenate([x1*c + x2*sn, x2*c - x1*sn], -1)
        return np.concatenate([x1*c - x2*sn, x2*c + x1*sn], -1)

class GeminiGQA:
    def __init__(self, d, h=8, g=2):
        self.d, self.h, self.g, self.hd = d, h, g, d//h
        self.wq, self.wk, self.wv, self.wo = Linear(d, d), Linear(d, d//g), Linear(d, d//g), Linear(d, d)
        self.rope, self.sc = RoPE(self.hd), (d//h)**-0.5

    def forward(self, x):
        b, s, _ = x.shape
        q = self.wq.forward(x).reshape(b, s, self.h, self.hd)
        k = self.wk.forward(x).reshape(b, s, self.h//self.g, self.hd)
        v = self.wv.forward(x).reshape(b, s, self.h//self.g, self.hd)
        self.qr, self.kr, self.v_raw = self.rope.apply(q), self.rope.apply(k), v
        ke, ve = np.repeat(self.kr, self.g, 2), np.repeat(v, self.g, 2)
        at = np.einsum("bshd,bthd->bsht", self.qr, ke) * self.sc
        at_max = at.max(-1, keepdims=True)
        exp_at = np.exp(at - at_max)
        self.p = exp_at / (exp_at.sum(-1, keepdims=True) + 1e-12)
        out = np.einsum("bsht,bthd->bshd", self.p, ve).reshape(b, s, -1)
        return self.wo.forward(out)

    def backward(self, dy):
        b, s, _ = dy.shape
        dy_wo = self.wo.backward(dy).reshape(b, s, self.h, self.hd)
        ke, ve = np.repeat(self.kr, self.g, 2), np.repeat(self.v_raw, self.g, 2)
        dp = np.einsum("bshd,bthd->bsht", dy_wo, ve)
        da = self.p * (dp - (self.p * dp).sum(-1, keepdims=True)) * self.sc
        dq = self.rope.apply(np.einsum("bsht,bthd->bshd", da, ke), True)
        dk_e = np.einsum("bsht,bshd->bthd", da, self.qr)
        dv_e = np.einsum("bsht,bshd->bthd", self.p, dy_wo)
        dk = self.rope.apply(dk_e.reshape(b, s, self.h//self.g, self.g, self.hd).sum(3), True)
        dv = dv_e.reshape(b, s, self.h//self.g, self.g, self.hd).sum(3)
        return self.wq.backward(dq.reshape(b, s, -1)) + self.wk.backward(dk.reshape(b, s, -1)) + self.wv.backward(dv.reshape(b, s, -1))

class GroqMoE:
    def __init__(self, d, n=4, e=2):
        self.n, self.d = n, d
        self.gate = Linear(d, n)
        self.experts = [[Linear(d, d*e), Linear(d*e, d), Linear(d, d*e)] for _ in range(n)]

    def forward(self, x):
        self.x = x
        g_lg = self.gate.forward(x)
        ex_g = np.exp(g_lg - g_lg.max(-1, keepdims=True))
        self.pr = ex_g / ex_g.sum(-1, keepdims=True)
        self.eo = []
        out = np.zeros_like(x)
        for i in range(self.n):
            w1, w2, w3 = self.experts[i]
            x1, x3 = w1.forward(x), w3.forward(x)
            sig = 1 / (1 + np.exp(-np.clip(x1, -12, 12)))
            sw = x1 * sig
            o = w2.forward(sw * x3)
            self.eo.append((x1, x3, sig, sw, o))
            out += self.pr[..., i:i+1] * o
        return out

    def backward(self, dy):
        dx = np.zeros_like(self.x)
        dp = np.zeros_like(self.pr)
        for i in range(self.n):
            x1, x3, sig, sw, o = self.eo[i]
            w1, w2, w3 = self.experts[i]
            dp[..., i] = (dy * o).sum(-1)
            dy_exp = dy * self.pr[..., i:i+1]
            dw2 = w2.backward(dy_exp)
            dx3, dsw = dw2 * sw, dw2 * x3
            dx1 = dsw * (sig + x1 * sig * (1 - sig))
            dx += w1.backward(dx1) + w3.backward(dx3)
        dl = self.pr * (dp - (self.pr * dp).sum(-1, keepdims=True))
        return dx + self.gate.backward(dl)

class UnifiedAttention:
    def __init__(self, d, h=8, g=2, n=4, e=2):
        self.d, self.h, self.g, self.hd = d, h, g, d//h
        self.wq, self.wk, self.wv, self.wo = Linear(d, d), Linear(d, d//g), Linear(d, d//g), Linear(d, d)
        self.rope, self.sc = RoPE(self.hd), (d//h)**-0.5
        self.gate = Linear(d, n)
        self.experts = [[Linear(d, d*e), Linear(d*e, d), Linear(d, d*e)] for _ in range(n)]

    def forward(self, x):
        b, s, _ = x.shape
        q = self.wq.forward(x).reshape(b, s, self.h, self.hd)
        k = self.wk.forward(x).reshape(b, s, self.h//self.g, self.hd)
        v = self.wv.forward(x).reshape(b, s, self.h//self.g, self.hd)
        self.qr, self.kr, self.v_raw = self.rope.apply(q), self.rope.apply(k), v
        ke, ve = np.repeat(self.kr, self.g, 2), np.repeat(v, self.g, 2)
        at = np.einsum("bshd,bthd->bsht", self.qr, ke) * self.sc
        at_max = at.max(-1, keepdims=True)
        exp_at = np.exp(at - at_max)
        self.p = exp_at / (exp_at.sum(-1, keepdims=True) + 1e-12)
        out = np.einsum("bsht,bthd->bshd", self.p, ve).reshape(b, s, -1)
        g_lg = self.gate.forward(x)
        ex_g = np.exp(g_lg - g_lg.max(-1, keepdims=True))
        self.pr = ex_g / ex_g.sum(-1, keepdims=True)
        self.eo = []
        out2 = np.zeros_like(x)
        for i in range(len(self.experts)):
            w1, w2, w3 = self.experts[i]
            x1, x3 = w1.forward(x), w3.forward(x)
            sig = 1 / (1 + np.exp(-np.clip(x1, -12, 12)))
            sw = x1 * sig
            o = w2.forward(sw * x3)
            self.eo.append((x1, x3, sig, sw, o))
            out2 += self.pr[..., i:i+1] * o
        return self.wo.forward(out) + out2

    def backward(self, dy):
        b, s, _ = dy.shape
        dy_wo = self.wo.backward(dy).reshape(b, s, self.h, self.hd)
        ke, ve = np.repeat(self.kr, self.g, 2), np.repeat(self.v_raw, self.g, 2)
        dp = np.einsum("bshd,bthd->bsht", dy_wo, ve)
        da = self.p * (dp - (self.p * dp).sum(-1, keepdims=True)) * self.sc
        dq = self.rope.apply(np.einsum("bsht,bthd->bshd", da, ke), True)
        dk_e = np.einsum("bsht,bshd->bthd", da, self.qr)
        dv_e = np.einsum("bsht,bshd->bthd", self.p, dy_wo)
        dk = self.rope.apply(dk_e.reshape(b, s, self.h//self.g, self.g, self.hd).sum(3), True)
        dv = dv_e.reshape(b, s, self.h//self.g, self.g, self.hd).sum(3)
        dx = self.wq.backward(dq.reshape(b, s, -1)) + self.wk.backward(dk.reshape(b, s, -1)) + self.wv.backward(dv.reshape(b, s, -1))
        dp = np.zeros_like(self.pr)
        for i in range(len(self.experts)):
            x1, x3, sig, sw, o = self.eo[i]
            w1, w2, w3 = self.experts[i]
            dp[..., i] = (dy * o).sum(-1)
            dy_exp = dy * self.pr[..., i:i+1]
            dw2 = w2.backward(dy_exp)
            dx3, dsw = dw2 * sw, dw2 * x3
            dx1 = dsw * (sig + x1 * sig * (1 - sig))
            dx += w1.backward(dx1) + w3.backward(dx3)
        dl = self.pr * (dp - (self.pr * dp).sum(-1, keepdims=True))
        return dx + self.gate.backward(dl)

class SovereignFusion:
    def __init__(self, d):
        self.unified = UnifiedAttention(d)

    def forward(self, x):
        return self.unified.forward(x)

    def backward(self, dy):
        return self.unified.backward(dy)

class SovereignBlock:
    def __init__(self, d):
        self.n1, self.f1, self.n2, self.f2 = RMSNorm(d), SovereignFusion(d), RMSNorm(d), UnifiedAttention(d)

    def forward(self, x):
        x = x + self.f1.forward(self.n1.forward(x))
        return x + self.f2.forward(self.n2.forward(x))

    def backward(self, dy):
        dy = dy + self.n2.backward(self.f2.backward(dy))
        return dy + self.n1.backward(self.f1.backward(dy))

class OMEGA_ASI:
    def __init__(self, i, h, o, d=2):
        self.emb = Linear(i, h)
        self.blks = [SovereignBlock(h) for _ in range(d)]
        self.ln = RMSNorm(h)
        self.head = Linear(h, o)

    def forward(self, x):
        if x.ndim == 2: x = x[:, None]
        x = self.emb.forward(x)
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
        def g(obj):
            if isinstance(obj, (Linear, RMSNorm)): p.append(obj)
            elif isinstance(obj, list): [g(i) for i in obj]
            elif hasattr(obj, "__dict__"): [g(v) for k, v in obj.__dict__.items() if k[0]!='_']
        g(self); return list(set(p))

class AdamW:
    def __init__(self, p, lr=1e-3, b1=.9, b2=.99, wd=.01):
        self.p, self.lr, self.b1, self.b2, self.wd, self.t = p, lr, b1, b2, wd, 0
        self.m = {id(x): [np.zeros_like(getattr(x, a)) for a in (['W','b'] if hasattr(x,'W') else ['g'])] for x in p}
        self.v = {id(x): [np.zeros_like(getattr(x, a)) for a in (['W','b'] if hasattr(x,'W') else ['g'])] for x in p}

    def step(self):
        self.t += 1
        lr_t = self.lr * min(1.0, self.t / 100) * (0.5 * (1 + np.cos(min(self.t, 1000) * np.pi / 1000)))
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
    N_SAMP, DIM, CATS, BS, EPS = 1024, 784, 10, 64, 100
    X = np.random.randn(N_SAMP, DIM).astype('f4')
    Y = np.random.randint(0, CATS, N_SAMP)
    model = OMEGA_ASI(DIM, 128, CATS, d=2)
    ps = model.params()
    opt = AdamW(ps, lr=2e-3)

    for e in range(EPS):
        idx = np.random.permutation(N_SAMP)
        l_sum, a_sum = 0, 0
        for i in range(0, N_SAMP, BS):
            xb, yb = X[idx[i:i+BS]], Y[idx[i:i+BS]]
            logits = model.forward(xb)
            probs = np.exp(logits - logits.max(-1, keepdims=True))
            probs /= probs.sum(-1, keepdims=True)
            l_sum += -np.log(probs[range(len(yb)), yb] + 1e-12).sum()
            a_sum += (probs.argmax(1) == yb).sum()
            dy = probs.copy()
            dy[range(len(yb)), yb] -= 1
            model.backward(dy / len(yb))

            gn = np.sqrt(sum((getattr(p,'d'+a if a!='g' else 'dg')**2).sum() for p in ps for a in (['W','b'] if hasattr(p,'W') else ['g'])))
            if gn > 1.0:
                for p in ps:
                    for a in (['W','b'] if hasattr(p,'W') else ['g']):
                        k = 'd'+a if a!='g' else 'dg'
                        setattr(p, k, getattr(p, k) / gn)
            opt.step()
        if (e+1) % 10 == 0:
            print(f"EVO {e+1:03d} | LOSS: {l_sum/N_SAMP:.4f} | ACC: {a_sum/N_SAMP:.4f} | GNORM: {gn:.2f}")

if __name__ == "__main__":
    train()
