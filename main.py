import numpy as np
import time

def swiglu(x):
    h = x.shape[-1] // 2
    a, b = x[..., :h], x[..., h:]
    return (a / (1.0 + np.exp(-np.clip(a, -12, 12)))) * b

def d_swiglu(x, dout):
    h = x.shape[-1] // 2
    a, b = x[..., :h], x[..., h:]
    s = 1.0 / (1.0 + np.exp(-np.clip(a, -12, 12)))
    sw = a * s
    da = dout * b * (s + sw * (1.0 - s))
    db = dout * sw
    return np.concatenate([da, db], axis=-1)

class Linear:
    def __init__(self, in_d, out_d, k=True):
        sc = np.sqrt(2./in_d) if k else 0.02
        self.W = np.random.normal(0, sc, (in_d, out_d)).astype(np.float32)
        self.b = np.zeros(out_d, dtype=np.float32)
        self.x, self.dW, self.db = None, None, None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        xf, df = self.x.reshape(-1, self.x.shape[-1]), dout.reshape(-1, dout.shape[-1])
        self.dW, self.db = xf.T @ df, np.sum(df, axis=0)
        return dout @ self.W.T

class RMSNorm:
    def __init__(self, dim, eps=1e-6):
        self.g, self.eps = np.ones(dim, dtype=np.float32), eps

    def forward(self, x):
        self.x = x
        self.ms = np.mean(x**2, axis=-1, keepdims=True)
        self.inv = 1.0 / np.sqrt(self.ms + self.eps)
        return self.g * (x * self.inv)

    def backward(self, dout):
        nx = self.x * self.inv
        self.dg = np.sum(dout * nx, axis=tuple(range(len(dout.shape)-1)))
        dnx = dout * self.g
        return self.inv * (dnx - nx * np.mean(dnx * nx, axis=-1, keepdims=True))

class RoPE:
    def __init__(self, dim, max_seq=2048):
        f = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
        emb = np.outer(np.arange(max_seq), f)
        emb = np.concatenate([emb, emb], axis=-1)
        self.cos, self.sin = np.cos(emb)[None,:,None,:], np.sin(emb)[None,:,None,:]

    def apply(self, x, rev=False):
        s, h = x.shape[1], x.shape[-1] // 2
        c, sn = self.cos[:,:s], self.sin[:,:s]
        xr = np.concatenate([-x[...,h:], x[...,:h]], axis=-1) if not rev else np.concatenate([x[...,h:], -x[...,:h]], axis=-1)
        return x * c + xr * sn

class Attention:
    def __init__(self, dim, heads=8, kv_heads=2):
        self.d, self.h, self.kv, self.hd = dim, heads, kv_heads, dim // heads
        self.g = heads // kv_heads
        self.q_p, self.k_p, self.v_p, self.o_p = Linear(dim, dim), Linear(dim, kv_heads*self.hd), Linear(dim, kv_heads*self.hd), Linear(dim, dim)
        self.rope, self.sc = RoPE(self.hd), 1.0 / np.sqrt(self.hd)

    def forward(self, x):
        b, s, _ = x.shape
        self.q = self.q_p.forward(x).reshape(b, s, self.h, self.hd)
        self.k = self.k_p.forward(x).reshape(b, s, self.kv, self.hd)
        self.v = self.v_p.forward(x).reshape(b, s, self.kv, self.hd)
        qr, kr = self.rope.apply(self.q), self.rope.apply(self.k)
        self.krp, self.vrp = np.repeat(kr, self.g, 2), np.repeat(self.v, self.g, 2)
        at = np.einsum("bshd,bthd->bsht", qr, self.krp) * self.sc
        e = np.exp(at - np.max(at, -1, keepdims=True))
        self.p = e / (np.sum(e, -1, keepdims=True) + 1e-12)
        out = np.einsum("bsht,bthd->bshd", self.p, self.vrp)
        return self.o_p.forward(out.reshape(b, s, self.d))

    def backward(self, dout):
        b, s, d = dout.shape
        do = self.o_p.backward(dout).reshape(b, s, self.h, self.hd)
        dvp, dp = np.einsum("bsht,bshd->bthd", self.p, do), np.einsum("bshd,bthd->bsht", do, self.vrp)
        da = self.p * (dp - np.sum(self.p * dp, -1, keepdims=True)) * self.sc
        dqr, dkrp = np.einsum("bsht,bthd->bshd", da, self.krp), np.einsum("bsht,bshd->bthd", da, self.rope.apply(self.q))
        dq, dk = self.rope.apply(dqr, True), self.rope.apply(np.sum(dkrp.reshape(b,s,self.kv,self.g,self.hd), 3), True)
        dv = np.sum(dvp.reshape(b,s,self.kv,self.g,self.hd), 3)
        return self.q_p.backward(dq.reshape(b,s,-1)) + self.k_p.backward(dk.reshape(b,s,-1)) + self.v_p.backward(dv.reshape(b,s,-1))

class Consensus:
    def __init__(self, dim):
        self.mlp = [Linear(dim, dim*4), Linear(dim*2, dim)]
        self.lin, self.gate = Linear(dim, dim), Linear(dim, 2)

    def forward(self, x):
        self.x, self.gh = x, swiglu(self.mlp[0].forward(x))
        self.go, self.qo = self.mlp[1].forward(self.gh), self.lin.forward(x)
        l = self.gate.forward(x)
        e = np.exp(l - np.max(l, -1, keepdims=True))
        self.p = e / (np.sum(e, -1, keepdims=True) + 1e-12)
        return self.p[...,0:1] * self.go + self.p[...,1:2] * self.qo

    def backward(self, dout):
        dp = np.concatenate([np.sum(dout * self.go, -1, keepdims=True), np.sum(dout * self.qo, -1, keepdims=True)], -1)
        dl = self.p * (dp - np.sum(self.p * dp, -1, keepdims=True))
        dx_g = self.mlp[0].backward(d_swiglu(self.mlp[0].x, self.mlp[1].backward(dout * self.p[...,0:1])))
        return dx_g + self.lin.backward(dout * self.p[...,1:2]) + self.gate.backward(dl)

class MoE:
    def __init__(self, dim, n=4, m=2):
        self.dim, self.n, self.gate = dim, n, Linear(dim, n)
        self.exp = [[Linear(dim, dim*m), Linear(dim*m, dim)] for _ in range(n)]

    def forward(self, x):
        self.sh = x.shape
        xf = x.reshape(-1, self.dim)
        l = self.gate.forward(xf)
        e = np.exp(l - np.max(l, -1, keepdims=True))
        self.p, self.idx = e / (np.sum(e, -1, keepdims=True) + 1e-12), np.argmax(e, -1)
        out, self.ca = np.zeros_like(xf), []
        for i in range(self.n):
            mk = self.idx == i
            if not np.any(mk): self.ca.append(None); continue
            h = self.exp[i][0].forward(xf[mk])
            act = swiglu(h)
            eo = self.exp[i][1].forward(act)
            out[mk] = eo * self.p[mk, i][:, None]
            self.ca.append((mk, h, eo))
        return out.reshape(self.sh)

    def backward(self, dout):
        df, dl, dx = dout.reshape(-1, self.dim), np.zeros_like(self.p), np.zeros((np.prod(self.sh[:-1]), self.dim))
        for i in range(self.n):
            if self.ca[i] is None: continue
            mk, h, eo = self.ca[i]
            dl[mk, i] = np.sum(df[mk] * eo, -1)
            dx[mk] += self.exp[i][0].backward(d_swiglu(h, self.exp[i][1].backward(df[mk] * self.p[mk, i][:,None])))
        dg = self.p * (dl - np.sum(self.p * dl, -1, keepdims=True))
        return (dx + self.gate.backward(dg)).reshape(self.sh)

class Block:
    def __init__(self, dim):
        self.n1, self.at = RMSNorm(dim), Attention(dim)
        self.n2, self.cn = RMSNorm(dim), Consensus(dim)
        self.n3, self.mo = RMSNorm(dim), MoE(dim)

    def forward(self, x):
        x = x + self.at.forward(self.n1.forward(x))
        x = x + self.cn.forward(self.n2.forward(x))
        return x + self.mo.forward(self.n3.forward(x))

    def backward(self, dout):
        dx = dout + self.n3.backward(self.mo.backward(dout))
        dx = dx + self.n2.backward(self.cn.backward(dx))
        return dx + self.n1.backward(self.at.backward(dx))

class OMEGA_ASI_X4:
    def __init__(self, in_d=784, h_d=64, out_d=10, depth=1):
        self.stem = Linear(in_d, h_d)
        self.blks = [Block(h_d) for _ in range(depth)]
        self.norm, self.head = RMSNorm(h_d), Linear(h_d, out_d)

    def forward(self, x):
        x = self.stem.forward(x)[:, None, :]
        for b in self.blks: x = b.forward(x)
        self.f = self.norm.forward(x[:, 0, :])
        return self.head.forward(self.f)

    def backward(self, dout):
        dout = self.norm.backward(self.head.backward(dout))[:, None, :]
        for b in reversed(self.blks): dout = b.backward(dout)
        self.stem.backward(dout[:, 0, :])

    def params(self):
        ps = []
        def coll(o):
            if isinstance(o, (Linear, RMSNorm)): ps.append(o)
            elif isinstance(o, list): [coll(i) for i in o]
            elif hasattr(o, "__dict__"): [coll(v) for v in o.__dict__.values()]
        coll(self)
        return ps

class Lion:
    def __init__(self, ps, lr=1e-4, b1=0.9, b2=0.99, wd=0.01):
        self.ps, self.lr, self.b1, self.b2, self.wd = ps, lr, b1, b2, wd
        self.m = [np.zeros_like(getattr(p, 'W', getattr(p, 'g', 0))) for p in ps]
        self.mb = [np.zeros_like(p.b) if hasattr(p, 'b') else None for p in ps]

    def step(self):
        for i, p in enumerate(self.ps):
            if hasattr(p, 'W'):
                for at, ml in [('W', self.m), ('b', self.mb)]:
                    if ml[i] is None: continue
                    g, m, w = getattr(p, 'd'+at), ml[i], getattr(p, at)
                    u = np.sign(self.b1 * m + (1.-self.b1) * g)
                    w -= self.lr * (u + (self.wd * w if at == 'W' else 0))
                    ml[i] = self.b2 * m + (1.-self.b2) * g
                    setattr(p, at, w)
            else:
                u = np.sign(self.b1 * self.m[i] + (1.-self.b1) * p.dg)
                p.g -= self.lr * (u + self.wd * p.g)
                self.m[i] = self.b2 * self.m[i] + (1.-self.b2) * p.dg

def train():
    X = np.random.randn(1024, 784).astype(np.float32)
    y = np.random.randint(0, 10, 1024)
    model = OMEGA_ASI_X4()
    ps = model.params()
    opt = Lion(ps)
    for ep in range(10):
        idx = np.random.permutation(1024)
        ls, acc, t0 = 0, 0, time.time()
        for i in range(0, 1024, 32):
            xb, yb = X[idx[i:i+32]], y[idx[i:i+32]]
            logits = model.forward(xb)
            probs = np.exp(logits - np.max(logits, 1, keepdims=True))
            probs /= np.sum(probs, 1, keepdims=True) + 1e-12
            ls += -np.mean(np.log(probs[range(len(yb)), yb] + 1e-10)) * len(yb)
            acc += np.sum(np.argmax(probs, 1) == yb)
            dout = probs.copy(); dout[range(len(yb)), yb] -= 1
            model.backward(dout / len(yb))
            gn = np.sqrt(sum(np.sum(getattr(p, 'dW', 0)**2) + np.sum(getattr(p, 'db', 0)**2) + np.sum(getattr(p, 'dg', 0)**2) for p in ps))
            if gn > 1.0:
                for p in ps:
                    if hasattr(p, 'dW'): p.dW /= gn; p.db /= gn
                    if hasattr(p, 'dg'): p.dg /= gn
            opt.step()
        print(f"EP:{ep} | LOSS:{ls/1024:.4f} | ACC:{acc/1024:.4f}")

if __name__ == "__main__": train()