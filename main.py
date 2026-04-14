import numpy as np

def silu(x):
    return x * (1 / (1 + np.exp(-np.clip(x, -10, 10))))

def dsilu(x, d):
    s = 1 / (1 + np.exp(-np.clip(x, -10, 10)))
    return d * (s * (1 + x * (1 - s)))

class Linear:
    def __init__(self, i, o, s=1.0):
        self.W = np.random.randn(i, o).astype('f') * (np.sqrt(2 / i) * s)
        self.b = np.zeros(o, 'f')
        self.dW, self.db = None, None
    def forward(self, x):
        self.x = x
        return x @ self.W + self.b
    def backward(self, d):
        self.dW = self.x.reshape(-1, self.x.shape[-1]).T @ d.reshape(-1, d.shape[-1])
        self.db = d.sum(axis=tuple(range(d.ndim - 1)))
        return d @ self.W.T

class RMSNorm:
    def __init__(self, d, e=1e-6):
        self.g, self.e = np.ones(d, 'f'), e
    def forward(self, x):
        self.x = x
        self.v = 1 / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.e)
        return self.g * (x * self.v)
    def backward(self, d):
        nx = self.x * self.v
        self.dg = (d * nx).sum(axis=tuple(range(d.ndim - 1)))
        dn = d * self.g
        return self.v * (dn - nx * np.mean(dn * nx, axis=-1, keepdims=True))

class RoPE:
    def __init__(self, d, m=2048):
        f = 1 / (10000**(np.arange(0, d, 2) / d))
        t = np.arange(m)
        fr = np.outer(t, f)
        self.c, self.s = np.cos(fr)[None, :, None, :], np.sin(fr)[None, :, None, :]
    def apply(self, x, r=False):
        b, s, h, d = x.shape
        d2 = d // 2
        x1, x2 = x[..., :d2], x[..., d2:]
        c, sn = self.c[:, :s], self.s[:, :s]
        return np.concatenate([x1 * c + x2 * sn, x2 * c - x1 * sn], -1) if r else np.concatenate([x1 * c - x2 * sn, x2 * c + x1 * sn], -1)

class GQA:
    def __init__(self, d, h=8, k=2):
        self.d, self.h, self.k, self.hd = d, h, k, d // h
        self.g = h // k
        self.wq, self.wk, self.wv, self.wo = Linear(d, d), Linear(d, k * self.hd), Linear(d, k * self.hd), Linear(d, d)
        self.rope, self.sc = RoPE(self.hd), (d // h)**-0.5
    def forward(self, x):
        b, s, _ = x.shape
        q = self.wq.forward(x).reshape(b, s, self.h, self.hd)
        k = self.wk.forward(x).reshape(b, s, self.k, self.hd)
        v = self.wv.forward(x).reshape(b, s, self.k, self.hd)
        self.q, self.k, self.v = self.rope.apply(q), self.rope.apply(k), v
        kr, vr = np.repeat(self.k, self.g, 2), np.repeat(v, self.g, 2)
        at = np.einsum("bshd,bthd->bsht", self.q, kr) * self.sc
        ex = np.exp(at - np.max(at, axis=-1, keepdims=True))
        self.p = ex / (np.sum(ex, axis=-1, keepdims=True) + 1e-10)
        o = np.einsum("bsht,bthd->bshd", self.p, vr)
        return self.wo.forward(o.reshape(b, s, self.d))
    def backward(self, d):
        b, s, _ = d.shape
        do = self.wo.backward(d).reshape(b, s, self.h, self.hd)
        kr, vr = np.repeat(self.k, self.g, 2), np.repeat(self.v, self.g, 2)
        dvr = np.einsum("bsht,bshd->bthd", self.p, do)
        dp = np.einsum("bshd,bthd->bsht", do, vr)
        da = self.p * (dp - np.sum(self.p * dp, axis=-1, keepdims=True)) * self.sc
        dq = self.rope.apply(np.einsum("bsht,bthd->bshd", da, kr), True)
        dkr = np.einsum("bsht,bshd->bthd", da, self.q)
        dk = self.rope.apply(dkr.reshape(b, s, self.k, self.g, self.hd).sum(3), True)
        dv = dvr.reshape(b, s, self.k, self.g, self.hd).sum(3)
        return self.wq.backward(dq.reshape(b, s, -1)) + self.wk.backward(dk.reshape(b, s, -1)) + self.wv.backward(dv.reshape(b, s, -1))

class SovereignLogic:
    def __init__(self, d):
        self.gemini_path = [Linear(d, d * 2), Linear(d * 2, d)]
        self.groq_path = Linear(d, d)
        self.gate = Linear(d, 2)
    def forward(self, x):
        self.h_gem = silu(self.gemini_path[0].forward(x))
        self.o_gem = self.gemini_path[1].forward(self.h_gem)
        self.o_groq = self.groq_path.forward(x)
        g = self.gate.forward(x)
        ex = np.exp(g - np.max(g, axis=-1, keepdims=True))
        self.p = ex / (np.sum(ex, axis=-1, keepdims=True) + 1e-10)
        return self.p[..., :1] * self.o_gem + self.p[..., 1:2] * self.o_groq
    def backward(self, d):
        dp = np.stack([np.sum(d * self.o_gem, axis=-1), np.sum(d * self.o_groq, axis=-1)], -1)
        dg = self.p * (dp - np.sum(self.p * dp, axis=-1, keepdims=True))
        dx = self.gate.backward(dg)
        dgem = self.gemini_path[1].backward(d * self.p[..., :1])
        dx += self.gemini_path[0].backward(dsilu(self.gemini_path[0].x, dgem))
        dx += self.groq_path.backward(d * self.p[..., 1:2])
        return dx

class MoE:
    def __init__(self, d, n=4, k=2):
        self.d, self.n, self.k = d, n, k
        self.gate = Linear(d, n)
        self.experts = [[Linear(d, d * 2), Linear(d * 2, d)] for _ in range(n)]
    def forward(self, x):
        s = x.shape; x = x.reshape(-1, self.d)
        g = self.gate.forward(x)
        ex = np.exp(g - np.max(g, axis=-1, keepdims=True))
        p = ex / (np.sum(ex, axis=-1, keepdims=True) + 1e-10)
        self.top_k_idx = np.argsort(p, axis=-1)[:, -self.k:]
        self.top_k_p = np.take_along_axis(p, self.top_k_idx, axis=-1)
        self.top_k_p /= np.sum(self.top_k_p, axis=-1, keepdims=True)
        out = np.zeros_like(x); self.expert_cache = []
        for i in range(self.n):
            mask = np.any(self.top_k_idx == i, axis=-1)
            if not np.any(mask): self.expert_cache.append(None); continue
            p_i = self.top_k_p[mask, np.where(self.top_k_idx[mask] == i)[1]][:, None]
            h = silu(self.experts[i][0].forward(x[mask]))
            y = self.experts[i][1].forward(h)
            out[mask] += y * p_i
            self.expert_cache.append((mask, h, y, p_i))
        self.p = p
        return out.reshape(s)
    def backward(self, d):
        s = d.shape; d = d.reshape(-1, self.d)
        dx, dp = np.zeros_like(d), np.zeros_like(self.p)
        for i in range(self.n):
            if self.expert_cache[i] is None: continue
            mask, h, y, p_i = self.expert_cache[i]
            dp[mask, i] = np.sum(d[mask] * y, axis=-1)
            dy = d[mask] * p_i
            dh = self.experts[i][1].backward(dy)
            dx[mask] += self.experts[i][0].backward(dsilu(self.experts[i][0].x, dh))
        dg = self.p * (dp - np.sum(self.p * dp, axis=-1, keepdims=True))
        return (dx + self.gate.backward(dg)).reshape(s)

class Block:
    def __init__(self, d):
        self.n1, self.at = RMSNorm(d), GQA(d)
        self.n2, self.lg = RMSNorm(d), SovereignLogic(d)
        self.n3, self.mo = RMSNorm(d), MoE(d)
    def forward(self, x):
        x = x + self.at.forward(self.n1.forward(x))
        x = x + self.lg.forward(self.n2.forward(x))
        return x + self.mo.forward(self.n3.forward(x))
    def backward(self, d):
        d = d + self.n3.backward(self.mo.backward(d))
        d = d + self.n2.backward(self.lg.backward(d))
        return d + self.n1.backward(self.at.backward(d))

class OMEGA_ASI:
    def __init__(self, i=784, h=128, o=10, d=2):
        self.embed = Linear(i, h)
        self.blocks = [Block(h) for _ in range(d)]
        self.norm = RMSNorm(h)
        self.head = Linear(h, o)
    def forward(self, x):
        x = self.embed.forward(x)[:, None, :]
        for b in self.blocks: x = b.forward(x)
        self.latent = self.norm.forward(x[:, 0, :])
        return self.head.forward(self.latent)
    def backward(self, d):
        d = self.norm.backward(self.head.backward(d))[:, None, :]
        for b in reversed(self.blocks): d = b.backward(d)
        self.embed.backward(d[:, 0, :])
    def params(self):
        p = []
        def find(obj):
            if isinstance(obj, (Linear, RMSNorm)): p.append(obj)
            elif isinstance(obj, list): [find(i) for i in obj]
            elif hasattr(obj, "__dict__"): [find(v) for v in obj.__dict__.values()]
        find(self); return p

class Lion:
    def __init__(self, p, lr=1e-4, b1=0.9, b2=0.99, wd=0.01):
        self.p, self.lr, self.b1, self.b2, self.wd = p, lr, b1, b2, wd
        self.m = [np.zeros_like(getattr(x, 'W', getattr(x, 'g', 0))) for x in p]
        self.mb = [np.zeros_like(x.b) if hasattr(x, 'b') else None for x in p]
    def step(self):
        for i, p in enumerate(self.p):
            if hasattr(p, 'W'):
                for attr, mom in [('W', self.m), ('b', self.mb)]:
                    if mom[i] is None: continue
                    g, w = getattr(p, 'd' + attr), getattr(p, attr)
                    u = np.sign(self.b1 * mom[i] + (1 - self.b1) * g)
                    w -= self.lr * (u + self.wd * w if attr == 'W' else u)
                    mom[i] = self.b2 * mom[i] + (1 - self.b2) * g
                    setattr(p, attr, w)
            else:
                u = np.sign(self.b1 * self.m[i] + (1 - self.b1) * p.dg)
                p.g -= self.lr * (u + self.wd * p.g)
                self.m[i] = self.b2 * self.m[i] + (1 - self.b2) * p.dg

def train():
    N, D, C = 1024, 784, 10
    X = np.random.randn(N, D).astype('f')
    Y = np.random.randint(0, C, N)
    model = OMEGA_ASI(D, 128, C, 2)
    opt = Lion(model.params(), lr=2e-4)
    
    for epoch in range(20):
        idx = np.random.permutation(N)
        total_loss, total_acc = 0, 0
        for i in range(0, N, 64):
            xb, yb = X[idx[i:i+64]], Y[idx[i:i+64]]
            logits = model.forward(xb)
            probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs /= np.sum(probs, axis=1, keepdims=True) + 1e-10
            
            loss = -np.mean(np.log(probs[range(len(yb)), yb] + 1e-10))
            total_loss += loss * len(yb)
            total_acc += np.sum(np.argmax(probs, axis=1) == yb)
            
            dout = probs.copy()
            dout[range(len(yb)), yb] -= 1
            model.backward(dout / len(yb))
            
            gn = np.sqrt(sum((getattr(p, 'dW', 0)**2).sum() + (getattr(p, 'db', 0)**2).sum() + (getattr(p, 'dg', 0)**2).sum() for p in model.params()))
            if gn > 1.0:
                for p in model.params():
                    if hasattr(p, 'dW'): p.dW /= gn; p.db /= gn
                    if hasattr(p, 'dg'): p.dg /= gn
            opt.step()
            
        print(f"Epoch {epoch} | Loss: {total_loss/N:.4f} | Acc: {total_acc/N:.4f}")

if __name__ == "__main__":
    train()
