import numpy as np

class Linear:
    def __init__(self, i, o, name=""):
        self.W = (np.random.randn(i, o) * np.sqrt(2/i)).astype('f4')
        self.b = np.zeros(o, 'f4')
        self.dW, self.db = None, None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dy):
        self.dW = self.x.reshape(-1, self.x.shape[-1]).T @ dy.reshape(-1, dy.shape[-1])
        self.db = dy.sum(axis=tuple(range(dy.ndim - 1)))
        return dy @ self.W.T

class RMSNorm:
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
        if conj: return np.concatenate([x1*c+x2*sn, x2*c-x1*sn], -1)
        return np.concatenate([x1*c-x2*sn, x2*c+x1*sn], -1)

class GQA:
    def __init__(self, d, h=8, g=2):
        self.d, self.h, self.g, self.hd = d, h, g, d // h
        self.wq, self.wk, self.wv, self.wo = Linear(d, d), Linear(d, d//g), Linear(d, d//g), Linear(d, d)
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
        dq = self.rope.apply(np.einsum("bsht,bthd->bshd", da, ke), True)
        dk_e = np.einsum("bsht,bshd->bthd", da, self.qr)
        dv_e = np.einsum("bsht,bshd->bthd", self.p, dy_wo)
        dk = self.rope.apply(dk_e.reshape(b, s, self.h//self.g, self.g, self.hd).sum(3), True)
        dv = dv_e.reshape(b, s, self.h//self.g, self.g, self.hd).sum(3)
        return self.wq.backward(dq.reshape(b, s, -1)) + self.wk.backward(dk.reshape(b, s, -1)) + self.wv.backward(dv.reshape(b, s, -1))

class SwiGLU:
    def __init__(self, d, e=4):
        self.w1, self.w2, self.w3 = Linear(d, d*e), Linear(d*e, d), Linear(d, d*e)

    def forward(self, x):
        self.x1, self.x3 = self.w1.forward(x), self.w3.forward(x)
        self.sig = 1 / (1 + np.exp(-np.clip(self.x1, -12, 12)))
        self.swish = self.x1 * self.sig
        return self.w2.forward(self.swish * self.x3)

    def backward(self, dy):
        d_w2 = self.w2.backward(dy)
        dx3 = d_w2 * self.swish
        dswish = d_w2 * self.x3
        dx1 = dswish * (self.sig + self.x1 * self.sig * (1 - self.sig))
        return self.w1.backward(dx1) + self.w3.backward(dx3)

class MoE:
    def __init__(self, d, n_e=4, e_dim=4):
        self.gate = Linear(d, n_e)
        self.experts = [SwiGLU(d, e_dim) for _ in range(n_e)]
        self.n_e = n_e

    def forward(self, x):
        self.b, self.s, self.d = x.shape
        logits = self.gate.forward(x)
        exp = np.exp(logits - logits.max(-1, keepdims=True))
        self.probs = exp / exp.sum(-1, keepdims=True)
        out = np.zeros_like(x)
        self.expert_outs = []
        for i in range(self.n_e):
            e_out = self.experts[i].forward(x)
            self.expert_outs.append(e_out)
            out += self.probs[..., i:i+1] * e_out
        return out

    def backward(self, dy):
        dx = np.zeros((self.b, self.s, self.d), 'f4')
        dprobs = np.zeros_like(self.probs)
        for i in range(self.n_e):
            dprobs[..., i] = (dy * self.expert_outs[i]).sum(-1)
            dx += self.experts[i].backward(dy * self.probs[..., i:i+1])
        dlogits = self.probs * (dprobs - (self.probs * dprobs).sum(-1, keepdims=True))
        dx += self.gate.backward(dlogits)
        return dx

class Block:
    def __init__(self, d):
        self.n1, self.at, self.n2, self.ff = RMSNorm(d), GQA(d), RMSNorm(d), MoE(d)

    def forward(self, x):
        x = x + self.at.forward(self.n1.forward(x))
        return x + self.ff.forward(self.n2.forward(x))

    def backward(self, dy):
        dy_ff = self.ff.backward(dy)
        dy = dy + self.n2.backward(dy_ff)
        dy_at = self.at.backward(dy)
        return dy + self.n1.backward(dy_at)

class OMEGA_ASI:
    def __init__(self, i, h, o, d=4):
        self.emb = Linear(i, h)
        self.blks = [Block(h) for _ in range(d)]
        self.ln = RMSNorm(h)
        self.head = Linear(h, o)

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
            if isinstance(o, (Linear, RMSNorm)): p.append(o)
            elif isinstance(o, (list, tuple)): [f(i) for i in o]
            elif hasattr(o, "__dict__"): [f(v) for k, v in o.__dict__.items() if k not in ('x', 'r', 'p', 'probs', 'expert_outs', 'x1', 'x3', 'sig', 'swish', 'qr', 'kr', 'v_v')]
        f(self); return list(set(p))

class AdamW:
    def __init__(self, p, lr=1e-3, b1=0.9, b2=0.999, wd=0.01):
        self.p, self.lr, self.b1, self.b2, self.wd, self.t = p, lr, b1, b2, wd, 0
        self.m = {id(x): [np.zeros_like(getattr(x, a)) for a in (['W', 'b'] if hasattr(x, 'W') else ['g'])] for x in p}
        self.v = {id(x): [np.zeros_like(getattr(x, a)) for a in (['W', 'b'] if hasattr(x, 'W') else ['g'])] for x in p}

    def step(self):
        self.t += 1
        for x in self.p:
            attrs = ['W', 'b'] if hasattr(x, 'W') else ['g']
            for i, a in enumerate(attrs):
                g = getattr(x, 'd'+a if a!='g' else 'dg')
                m, v = self.m[id(x)][i], self.v[id(x)][i]
                m[:] = self.b1 * m + (1 - self.b1) * g
                v[:] = self.b2 * v + (1 - self.b2) * (g**2)
                mh = m / (1 - self.b1**self.t)
                vh = v / (1 - self.b2**self.t)
                p_val = getattr(x, a)
                p_val -= self.lr * (mh / (np.sqrt(vh) + 1e-8) + self.wd * p_val)
                setattr(x, a, p_val)

def train():
    N, D, C, BS, EP = 2048, 784, 10, 128, 30
    X = (np.random.randn(N, D) * 0.01).astype('f4')
    Y = np.random.randint(0, C, N)
    model = OMEGA_ASI(D, 128, C, d=3)
    ps = model.params()
    opt = AdamW(ps, lr=1e-3)
    
    for e in range(EP):
        idx = np.random.permutation(N)
        l_sum, a_sum = 0, 0
        for i in range(0, N, BS):
            xb, yb = X[idx[i:i+BS]], Y[idx[i:i+BS]]
            lg = model.forward(xb)
            pr = np.exp(lg - lg.max(-1, keepdims=True))
            pr /= pr.sum(-1, keepdims=True)
            l_sum += -np.log(pr[range(len(yb)), yb] + 1e-12).sum()
            a_sum += (pr.argmax(1) == yb).sum()
            dy = pr.copy(); dy[range(len(yb)), yb] -= 1
            model.backward(dy / len(yb))
            
            gn = np.sqrt(sum((getattr(p, 'd'+a if a!='g' else 'dg')**2).sum() for p in ps for a in (['W', 'b'] if hasattr(p, 'W') else ['g'])))
            if gn > 1.0:
                for p in ps:
                    for a in (['W', 'b'] if hasattr(p, 'W') else ['g']):
                        k = 'd'+a if a!='g' else 'dg'
                        setattr(p, k, getattr(p, k)/gn)
            opt.step()
        print(f"Epoch {e+1:02d} | Loss: {l_sum/N:.4f} | Acc: {a_sum/N:.4f}")

if __name__ == "__main__":
    train()
