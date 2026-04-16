import numpy as np

def silu(x): return x / (1 + np.exp(-np.clip(x, -12, 12)))

class Linear:
    def __init__(self, i, o, s=None):
        self.W = np.random.randn(i, o).astype("f4") * (s or np.sqrt(2/i))
        self.b = np.zeros(o, "f4")
        self.dW, self.db, self.x = None, None, None
    def forward(self, x):
        self.x = x
        return x @ self.W + self.b
    def backward(self, dy):
        self.dW = self.x.reshape(-1, self.x.shape[-1]).T @ dy.reshape(-1, dy.shape[-1])
        self.db = dy.reshape(-1, dy.shape[-1]).sum(0)
        return (dy @ self.W.T).reshape(self.x.shape[:-1] + (self.W.shape[0],))

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
        self.dg = (dy * xn).reshape(-1, xn.shape[-1]).sum(0)
        dn = dy * self.g
        return self.r * (dn - xn * np.mean(dn * xn, -1, keepdims=True))

class RoPE:
    def __init__(self, d, m=4096):
        f = 1. / (10000 ** (np.arange(0, d, 2) / d))
        t = np.outer(np.arange(m), f)
        self.c, self.s = np.cos(t), np.sin(t)
    def apply(self, x, conj=False):
        s, d = x.shape[1], x.shape[-1]
        x1, x2 = x[..., :d//2], x[..., d//2:]
        c, sn = self.c[:s, None, :], self.s[:s, None, :]
        if conj: return np.concatenate([x1 * c + x2 * sn, x2 * c - x1 * sn], -1)
        return np.concatenate([x1 * c - x2 * sn, x2 * c + x1 * sn], -1)

class GQA:
    def __init__(self, d, h=8, g=2):
        self.d, self.h, self.g, self.hd = d, h, g, d // h
        self.wq, self.wk, self.wv, self.wo = Linear(d, d), Linear(d, (h//g)*self.hd), Linear(d, (h//g)*self.hd), Linear(d, d)
        self.rp, self.sc = RoPE(self.hd), (d//h)**-0.5
    def forward(self, x):
        b, s, _ = x.shape
        q, k, v = self.wq.forward(x).reshape(b,s,self.h,self.hd), self.wk.forward(x).reshape(b,s,self.h//self.g,self.hd), self.wv.forward(x).reshape(b,s,self.h//self.g,self.hd)
        self.qr, self.kr, self.v_raw = self.rp.apply(q), self.rp.apply(k), v
        ke, ve = np.repeat(self.kr, self.g, 2), np.repeat(v, self.g, 2)
        at = np.einsum("bshd,bthd->bsht", self.qr, ke) * self.sc
        at -= at.max(-1, keepdims=True)
        self.p = np.exp(at); self.p /= self.p.sum(-1, keepdims=True) + 1e-12
        return self.wo.forward(np.einsum("bsht,bthd->bshd", self.p, ve).reshape(b, s, -1))
    def backward(self, dy):
        b, s = dy.shape[:2]
        dy_o = self.wo.backward(dy).reshape(b, s, self.h, self.hd)
        ke, ve = np.repeat(self.kr, self.g, 2), np.repeat(self.v_raw, self.g, 2)
        dp = np.einsum("bshd,bthd->bsht", dy_o, ve)
        da = self.p * (dp - (self.p * dp).sum(-1, keepdims=True)) * self.sc
        dq = self.rp.apply(np.einsum("bsht,bthd->bshd", da, ke), True)
        dk = self.rp.apply(np.einsum("bsht,bshd->bthd", da, self.qr).reshape(b,s,self.h//self.g,self.g,self.hd).sum(3), True)
        dv = np.einsum("bsht,bshd->bthd", self.p, dy_o).reshape(b,s,self.h//self.g,self.g,self.hd).sum(3)
        return self.wq.backward(dq.reshape(b,s,-1)) + self.wk.backward(dk.reshape(b,s,-1)) + self.wv.backward(dv.reshape(b,s,-1))

class SwiGLU:
    def __init__(self, d, h):
        self.w1, self.w2, self.w3 = Linear(d, h), Linear(h, d), Linear(d, h)
    def forward(self, x):
        self.x1, self.x3 = self.w1.forward(x), self.w3.forward(x)
        self.sig = 1 / (1 + np.exp(-np.clip(self.x1, -12, 12)))
        self.act = (self.x1 * self.sig) * self.x3
        return self.w2.forward(self.act)
    def backward(self, dy):
        da = self.w2.backward(dy)
        dx1 = da * self.x3 * (self.sig * (1 + self.x1 * (1 - self.sig)))
        dx3 = da * (self.x1 * self.sig)
        return self.w1.backward(dx1) + self.w3.backward(dx3)

class MoE:
    def __init__(self, d, n=4, e=2):
        self.n, self.gate = n, Linear(d, n)
        self.experts = [SwiGLU(d, d*e) for _ in range(n)]
    def forward(self, x):
        self.x = x
        g = self.gate.forward(x)
        self.pr = np.exp(g - g.max(-1, keepdims=True)); self.pr /= self.pr.sum(-1, keepdims=True)
        self.ex_out = [exp.forward(x) for exp in self.experts]
        o = np.zeros_like(x)
        for i in range(self.n): o += self.pr[..., i:i+1] * self.ex_out[i]
        return o
    def backward(self, dy):
        dx, dpr = np.zeros_like(self.x), np.zeros_like(self.pr)
        for i in range(self.n):
            dpr[..., i] = (dy * self.ex_out[i]).sum(-1)
            dx += self.experts[i].backward(dy * self.pr[..., i:i+1])
        dg = self.pr * (dpr - (self.pr * dpr).sum(-1, keepdims=True))
        return dx + self.gate.backward(dg)

class RedundantBlock:
    def __init__(self, d):
        self.n1, self.gemini_attn = RMSNorm(d), GQA(d)
        self.n2, self.groq_dense = RMSNorm(d), MoE(d)
        self.alpha = np.array([0.5], dtype="f4")
        self.d_alpha = 0
    def forward(self, x):
        self.x_in = x
        self.g_out = self.gemini_attn.forward(self.n1.forward(x))
        self.q_out = self.groq_dense.forward(self.n2.forward(x))
        return x + self.alpha * self.g_out + (1 - self.alpha) * self.q_out
    def backward(self, dy):
        self.d_alpha = (dy * (self.g_out - self.q_out)).sum()
        dg = dy * self.alpha
        dq = dy * (1 - self.alpha)
        dx = dy + self.n1.backward(self.gemini_attn.backward(dg)) + self.n2.backward(self.groq_dense.backward(dq))
        return dx

class OMEGA_ASI:
    def __init__(self, i, h, o, d=2):
        self.embed = Linear(i, h)
        self.blocks = [RedundantBlock(h) for _ in range(d)]
        self.norm = RMSNorm(h)
        self.head = Linear(h, o)
    def forward(self, x):
        x = self.embed.forward(x[:, None] if x.ndim==2 else x)
        for b in self.blocks: x = b.forward(x)
        return self.head.forward(self.norm.forward(x[:, -1]))
    def backward(self, dy):
        dy = self.norm.backward(self.head.backward(dy))
        dys = np.zeros((dy.shape[0], 1, dy.shape[1]), "f4")
        dys[:, -1] = dy
        for b in reversed(self.blocks): dys = b.backward(dys)
        self.embed.backward(dys)
    def params(self):
        res = []
        def collect(obj):
            if isinstance(obj, (Linear, RMSNorm)): res.append(obj)
            elif isinstance(obj, list): [collect(i) for i in obj]
            elif hasattr(obj, "__dict__"): [collect(v) for k, v in obj.__dict__.items() if k[0] != "_"]
        collect(self)
        return list(set(res))

class Optimizer:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.99, wd=0.01):
        self.p, self.lr, self.b1, self.b2, self.wd, self.t = params, lr, b1, b2, wd, 0
        self.m = {id(p): [np.zeros_like(getattr(p, a)) for a in (["W","b"] if hasattr(p,"W") else ["g"])] for p in params}
        self.v = {id(p): [np.zeros_like(getattr(p, a)) for a in (["W","b"] if hasattr(p,"W") else ["g"])] for p in params}
    def step(self):
        self.t += 1
        curr_lr = self.lr * min(1.0, self.t / 50) * (0.5 * (1 + np.cos(min(self.t, 500) * np.pi / 500)))
        for p in self.p:
            attrs = ["W", "b"] if hasattr(p, "W") else ["g"]
            for i, a in enumerate(attrs):
                grad = getattr(p, "d"+a if a!="g" else "dg")
                m, v = self.m[id(p)], self.v[id(p)]
                m[i] = self.b1 * m[i] + (1 - self.b1) * grad
                v[i] = self.b2 * v[i] + (1 - self.b2) * (grad**2)
                mh = m[i] / (1 - self.b1**self.t)
                vh = v[i] / (1 - self.b2**self.t)
                val = getattr(p, a)
                val -= curr_lr * (mh / (np.sqrt(vh) + 1e-8) + self.wd * val)
                setattr(p, a, val)

def train():
    N, D, C, BS, E = 1024, 784, 10, 32, 50
    X = np.random.randn(N, D).astype("f4")
    Y = np.random.randint(0, C, N)
    model = OMEGA_ASI(D, 128, C, d=2)
    ps = model.params()
    opt = Optimizer(ps, lr=2e-3)
    
    for e in range(E):
        idx = np.random.permutation(N)
        loss, acc = 0, 0
        for i in range(0, N, BS):
            xb, yb = X[idx[i:i+BS]], Y[idx[i:i+BS]]
            logits = model.forward(xb)
            
            ex = np.exp(logits - logits.max(-1, keepdims=True))
            probs = ex / ex.sum(-1, keepdims=True)
            
            loss += -np.log(probs[range(len(yb)), yb] + 1e-12).sum()
            acc += (probs.argmax(-1) == yb).sum()
            
            dy = probs.copy()
            dy[range(len(yb)), yb] -= 1
            model.backward(dy / len(yb))
            
            gn = np.sqrt(sum((getattr(p, "d"+a if a!="g" else "dg")**2).sum() for p in ps for a in (["W","b"] if hasattr(p,"W") else ["g"])))
            if gn > 1.0:
                for p in ps:
                    for a in (["W","b"] if hasattr(p,"W") else ["g"]):
                        k = "d"+a if a!="g" else "dg"
                        setattr(p, k, getattr(p, k) / gn)
            opt.step()
            
        if (e+1) % 5 == 0:
            print(f"EPOCH {e+1:03d} | LOSS: {loss/N:.4f} | ACC: {acc/N:.4f}")

if __name__ == "__main__":
    train()
