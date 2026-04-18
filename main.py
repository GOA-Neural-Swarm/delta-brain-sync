import numpy as np

class T:
    def __init__(self, d):
        self.data = d.astype("f4")
        self.grad = np.zeros_like(self.data)

class M:
    def params(self):
        p = []
        for v in self.__dict__.values():
            if isinstance(v, T): p.append(v)
            elif isinstance(v, M): p.extend(v.params())
            elif isinstance(v, list):
                for i in v: p.extend(i.params() if isinstance(i, M) else ([i] if isinstance(i, T) else []))
        return p

class L(M):
    def __init__(self, i, o):
        self.w = T(np.random.randn(i, o) * np.sqrt(2/i))
        self.b = T(np.zeros(o))
    def f(self, x):
        self.x = x
        return x @ self.w.data + self.b.data
    def b(self, dy):
        xf = self.x.reshape(-1, self.x.shape[-1])
        df = dy.reshape(-1, dy.shape[-1])
        self.w.grad += xf.T @ df
        self.b.grad += df.sum(0)
        return (dy @ self.w.data.T).reshape(self.x.shape)

class RMS(M):
    def __init__(self, d, e=1e-6):
        self.g, self.e = T(np.ones(d)), e
    def f(self, x):
        self.x = x
        self.v = np.mean(x**2, -1, keepdims=True)
        self.i = (self.v + self.e)**-0.5
        self.nx = x * self.i
        return self.g.data * self.nx
    def b(self, dy):
        dg = dy * self.g.data
        self.g.grad += np.sum(dy * self.nx, axis=tuple(range(dy.ndim - 1)))
        return (dg - self.nx * np.mean(dg * self.nx, -1, keepdims=True)) * self.i

class SwiGLU(M):
    def f(self, x):
        self.x = x
        self.g, self.v = np.split(x, 2, -1)
        self.sig = 1 / (1 + np.exp(-np.clip(self.g, -12, 12)))
        self.swish = self.g * self.sig
        return self.swish * self.v
    def b(self, dy):
        ds = dy * self.v
        dv = dy * self.swish
        dg = ds * self.sig * (1 + self.g * (1 - self.sig))
        return np.concatenate([dg, dv], -1)

class MLA(M):
    def __init__(self, d, h=8, ld=32):
        self.d, self.h, self.ld, self.hd = d, h, ld, d // h
        self.w_dkv = L(d, ld)
        self.w_uq = L(d, h * self.hd)
        self.w_ukv = L(ld, h * self.hd)
        self.w_o = L(d, d)
        self.sc = self.hd**-0.5
    def f(self, x):
        b, s, _ = x.shape
        self.kv_lat = self.w_dkv.f(x)
        self.q = self.w_uq.f(x).reshape(b, s, self.h, self.hd)
        self.kv = self.w_ukv.f(self.kv_lat).reshape(b, s, self.h, self.hd)
        self.at = np.einsum("bshd,bthd->bsht", self.q, self.kv) * self.sc
        self.p = (e := np.exp(self.at - self.at.max(-1, keepdims=True))) / (e.sum(-1, keepdims=True) + 1e-12)
        self.o = np.einsum("bsht,bthd->bshd", self.p, self.kv).reshape(b, s, -1)
        return self.w_o.f(self.o)
    def b(self, dy):
        b, s = dy.shape[:2]
        do = self.w_o.b(dy).reshape(b, s, self.h, self.hd)
        dp = np.einsum("bshd,bthd->bsht", do, self.kv)
        da = self.p * (dp - (self.p * dp).sum(-1, keepdims=True)) * self.sc
        dq = np.einsum("bsht,bthd->bshd", da, self.kv)
        dkv = (np.einsum("bsht,bshd->bthd", da, self.q) + np.einsum("bsht,bshd->bthd", self.p, do))
        dx = self.w_uq.b(dq.reshape(b, s, -1))
        dkv_lat = self.w_ukv.b(dkv.reshape(b, s, -1))
        return dx + self.w_dkv.b(dkv_lat)

class MoE(M):
    def __init__(self, d, n=4, k=2):
        self.d, self.n, self.k = d, n, k
        self.gate = L(d, n)
        self.experts = [[L(d, d*2), SwiGLU(), L(d, d)] for _ in range(n)]
    def f(self, x):
        self.sh = x.shape
        xf = x.reshape(-1, self.d)
        g = self.gate.f(xf)
        self.p = (e := np.exp(g - g.max(-1, keepdims=True))) / (e.sum(-1, keepdims=True) + 1e-12)
        self.idx = np.argsort(self.p, -1)[:, -self.k:]
        self.w = np.take_along_axis(self.p, self.idx, -1)
        self.w /= self.w.sum(-1, keepdims=True) + 1e-12
        out, self.cache = np.zeros_like(xf), []
        for i in range(self.n):
            m = np.any(self.idx == i, -1)
            if not np.any(m): self.cache.append(None); continue
            ps = np.where(self.idx[m] == i)[1]
            h = self.experts[i][2].f(self.experts[i][1].f(self.experts[i][0].f(xf[m])))
            out[m] += h * self.w[m, ps][:, None]
            self.cache.append((m, ps, h))
        return out.reshape(self.sh)
    def b(self, dy):
        dyf = dy.reshape(-1, self.d)
        dx, dg = np.zeros((dyf.shape[0], self.d)), np.zeros((dyf.shape[0], self.n))
        for i in range(self.n):
            if self.cache[i] is not None:
                m, ps, h = self.cache[i]
                dg[m, i] = (dyf[m] * h).sum(-1)
                dh = dyf[m] * self.w[m, ps][:, None]
                dx[m] += self.experts[i][0].b(self.experts[i][1].b(self.experts[i][2].b(dh)))
        return (dx + self.gate.b(dg - (self.p * dg).sum(-1, keepdims=True))).reshape(self.sh)

class RedundantFusion(M):
    def __init__(self, d):
        self.groq_path = L(d, d)
        self.gemini_path = MoE(d)
        self.alpha = T(np.array([0.5]))
    def f(self, x):
        self.o_groq = self.groq_path.f(x)
        self.o_gemini = self.gemini_path.f(x)
        return self.alpha.data * self.o_groq + (1 - self.alpha.data) * self.o_gemini
    def b(self, dy):
        self.alpha.grad += np.sum(dy * (self.o_groq - self.o_gemini))
        return self.alpha.data * self.groq_path.b(dy) + (1 - self.alpha.data) * self.gemini_path.b(dy)

class Block(M):
    def __init__(self, d):
        self.n1, self.at = RMS(d), MLA(d)
        self.n2, self.ff = RMS(d), RedundantFusion(d)
    def f(self, x):
        x = x + self.at.f(self.n1.f(x))
        return x + self.ff.f(self.n2.f(x))
    def b(self, dy):
        dy = dy + self.ff.b(self.n2.b(dy))
        return dy + self.at.b(self.n1.b(dy))

class OMEGA(M):
    def __init__(self, di, dm, do, dp=2):
        self.emb = L(di, dm)
        self.blocks = [Block(dm) for _ in range(dp)]
        self.norm = RMS(dm)
        self.head = L(dm, do)
    def f(self, x):
        x = self.emb.f(x[:, None] if x.ndim == 2 else x)
        for b in self.blocks: x = b.f(x)
        return self.head.f(self.norm.f(x[:, -1]))
    def b(self, dy):
        dy = self.norm.b(self.head.b(dy))
        db = np.zeros((dy.shape[0], self.emb.x.shape[1], dy.shape[1]))
        db[:, -1] = dy
        for b in reversed(self.blocks): db = b.b(db)
        self.emb.b(db)

class AdamW:
    def __init__(self, p, lr=1e-3, b1=0.9, b2=0.99, wd=0.01):
        self.p, self.lr, self.b1, self.b2, self.wd, self.t = p, lr, b1, b2, wd, 0
        self.m = [np.zeros_like(i.data) for i in p]
        self.v = [np.zeros_like(i.data) for i in p]
    def step(self):
        self.t += 1
        a = self.lr * (np.sqrt(1 - self.b2**self.t) / (1 - self.b1**self.t))
        for i, pt in enumerate(self.p):
            g = np.clip(pt.grad, -10, 10)
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g**2)
            pt.data -= a * (self.m[i] / (np.sqrt(self.v[i]) + 1e-8) + self.wd * pt.data)
            pt.grad.fill(0)

def train():
    N, D, C, BS, E = 2048, 784, 10, 64, 100
    X = np.random.randn(N, D).astype("f4")
    Y = np.random.randint(0, C, N)
    model = OMEGA(D, 128, C, 2)
    opt = AdamW(model.params(), lr=3e-3)
    
    for e in range(E):
        idx = np.random.permutation(N)
        losses, accs = [], []
        for i in range(0, N, BS):
            xb, yb = X[idx[i:i+BS]], Y[idx[i:i+BS]]
            logits = model.f(xb)
            probs = (ex := np.exp(logits - logits.max(-1, keepdims=True))) / (ex.sum(-1, keepdims=True) + 1e-12)
            loss = -np.mean(np.log(probs[range(len(yb)), yb] + 1e-12))
            losses.append(loss)
            accs.append(np.mean(probs.argmax(-1) == yb))
            
            dl = probs.copy()
            dl[range(len(yb)), yb] -= 1
            model.b(dl / len(yb))
            opt.step()
            
        if (e + 1) % 10 == 0:
            print(f"Epoch {e+1:03} | Loss: {np.mean(losses):.4f} | Acc: {np.mean(accs):.4f}")

if __name__ == "__main__":
    train()
