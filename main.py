import numpy as np

class T:
    def __init__(self, d, n=""):
        self.data = np.ascontiguousarray(d.astype("f4"))
        self.grad = np.zeros_like(self.data)

class Module:
    def params(self):
        p = []
        for v in self.__dict__.values():
            if isinstance(v, T): p.append(v)
            elif isinstance(v, Module): p.extend(v.params())
            elif isinstance(v, list):
                for i in v:
                    if isinstance(i, (Module, T)): p.extend(i.params() if isinstance(i, Module) else [i])
        return p

class Linear(Module):
    def __init__(self, i, o, b=True):
        self.w = T(np.random.randn(i, o) * (2/i)**.5)
        self.b = T(np.zeros(o)) if b else None
    def f(self, x):
        self.x = x
        return x @ self.w.data + (self.b.data if self.b else 0)
    def b(self, dy):
        xf = self.x.reshape(-1, self.x.shape[-1])
        df = dy.reshape(-1, dy.shape[-1])
        self.w.grad += xf.T @ df
        if self.b: self.b.grad += df.sum(0)
        return (dy @ self.w.data.T).reshape(self.x.shape)

class RMSNorm(Module):
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

class SwiGLU(Module):
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

class GQA(Module):
    def __init__(self, d, h=8, g=2):
        self.d, self.h, self.g, self.hd = d, h, g, d // h
        self.wq, self.wk, self.wv, self.wo = Linear(d, d), Linear(d, (h//g)*self.hd), Linear(d, (h//g)*self.hd), Linear(d, d)
        self.sc = self.hd**-0.5
        self._cache_rope(2048)
    def _cache_rope(self, sz):
        f = 10000**-(np.arange(0, self.hd, 2)/self.hd)
        t = np.arange(sz)[:, None] * f
        self.cos, self.sin = np.cos(t), np.sin(t)
    def _rope(self, x, inv=False):
        s = x.shape[1]
        c, s_ = self.cos[:s, None, :], self.sin[:s, None, :] * (-1 if inv else 1)
        r, i = x[..., ::2], x[..., 1::2]
        o = np.empty_like(x)
        o[..., ::2], o[..., 1::2] = r*c - i*s_, r*s_ + i*c
        return o
    def f(self, x):
        b, s, _ = x.shape
        q = self.wq.f(x).reshape(b, s, self.h, self.hd)
        k = self.wk.f(x).reshape(b, s, self.h//self.g, self.hd)
        v = self.wv.f(x).reshape(b, s, self.h//self.g, self.hd)
        self.qr, self.kr, self.v_orig = self._rope(q), self._rope(k), v
        kr_up = np.repeat(self.kr, self.g, 2)
        vr_up = np.repeat(v, self.g, 2)
        at = np.einsum("bshd,bthd->bsht", self.qr, kr_up) * self.sc
        self.p = (e := np.exp(at - at.max(-1, keepdims=True))) / (e.sum(-1, keepdims=True) + 1e-12)
        return self.wo.f(np.einsum("bsht,bthd->bshd", self.p, vr_up).reshape(b, s, -1))
    def b(self, dy):
        b, s = dy.shape[:2]
        dy_wo = self.wo.b(dy).reshape(b, s, self.h, self.hd)
        kr_up = np.repeat(self.kr, self.g, 2)
        vr_up = np.repeat(self.v_orig, self.g, 2)
        dp = np.einsum("bshd,bthd->bsht", dy_wo, vr_up)
        da = self.p * (dp - (self.p * dp).sum(-1, keepdims=True)) * self.sc
        dqr = np.einsum("bsht,bthd->bshd", da, kr_up)
        dkr = np.einsum("bsht,bshd->bthd", da, self.qr).reshape(b, s, self.h//self.g, self.g, self.hd).sum(3)
        dvr = np.einsum("bsht,bshd->bthd", self.p, dy_wo).reshape(b, s, self.h//self.g, self.g, self.hd).sum(3)
        return self.wq.b(self._rope(dqr, True).reshape(b, s, -1)) + \
               self.wk.b(self._rope(dkr, True).reshape(b, s, -1)) + \
               self.wv.b(dvr.reshape(b, s, -1))

class RedundantExpert(Module):
    def __init__(self, d, n_exp=4, top_k=2):
        self.d, self.n_exp, self.top_k = d, n_exp, top_k
        self.gate = Linear(d, n_exp)
        self.experts = [[Linear(d, d*2), SwiGLU(), Linear(d, d)] for _ in range(n_exp)]
    def f(self, x):
        self.sh = x.shape
        xf = x.reshape(-1, self.d)
        lg = self.gate.f(xf)
        p = (e := np.exp(lg - lg.max(-1, keepdims=True))) / (e.sum(-1, keepdims=True) + 1e-12)
        self.idx = np.argsort(p, -1)[:, -self.top_k:]
        self.w = np.take_along_axis(p, self.idx, -1)
        self.w /= self.w.sum(-1, keepdims=True) + 1e-12
        out, self.cache = np.zeros_like(xf), []
        for i in range(self.n_exp):
            m = np.any(self.idx == i, -1)
            if not np.any(m): self.cache.append(None); continue
            pos = np.where(self.idx[m] == i)[1]
            h1 = self.experts[i][0].f(xf[m])
            h2 = self.experts[i][1].f(h1)
            h3 = self.experts[i][2].f(h2)
            out[m] += h3 * self.w[m, pos][:, None]
            self.cache.append((m, pos, h3))
        return out.reshape(self.sh)
    def b(self, dy):
        dyf = dy.reshape(-1, self.d)
        dx, dg = np.zeros((dyf.shape[0], self.d)), np.zeros((dyf.shape[0], self.n_exp))
        for i in range(self.n_exp):
            if self.cache[i] is None: continue
            m, pos, h3 = self.cache[i]
            dg[m, i] = (dyf[m] * h3).sum(-1)
            dx[m] += self.experts[i][0].b(self.experts[i][1].b(self.experts[i][2].b(dyf[m] * self.w[m, pos][:, None])))
        return (dx + self.gate.b(dg - dg.mean(-1, keepdims=True))).reshape(self.sh)

class DualStreamLogic(Module):
    def __init__(self, d):
        self.gemini_path = [Linear(d, d*2), SwiGLU(), Linear(d, d)]
        self.groq_path = [Linear(d, d*2), SwiGLU(), Linear(d, d)]
        self.fusion = Linear(d*2, d)
    def f(self, x):
        self.x = x
        self.o1 = self.gemini_path[2].f(self.gemini_path[1].f(self.gemini_path[0].f(x)))
        self.o2 = self.groq_path[2].f(self.groq_path[1].f(self.groq_path[0].f(x)))
        return self.fusion.f(np.concatenate([self.o1, self.o2], -1))
    def b(self, dy):
        df = self.fusion.b(dy)
        do1, do2 = np.split(df, 2, -1)
        dx1 = self.gemini_path[0].b(self.gemini_path[1].b(self.gemini_path[2].b(do1)))
        dx2 = self.groq_path[0].b(self.groq_path[1].b(self.groq_path[2].b(do2)))
        return dx1 + dx2

class Block(Module):
    def __init__(self, d):
        self.n1, self.at = RMSNorm(d), GQA(d)
        self.n2, self.ds = RMSNorm(d), DualStreamLogic(d)
        self.n3, self.moe = RMSNorm(d), RedundantExpert(d)
    def f(self, x):
        x = x + self.at.f(self.n1.f(x))
        x = x + self.ds.f(self.n2.f(x))
        return x + self.moe.f(self.n3.f(x))
    def b(self, dy):
        dy = dy + self.moe.b(self.n3.b(dy))
        dy = dy + self.ds.b(self.n2.b(dy))
        return dy + self.at.b(self.n1.b(dy))

class OMEGA_ASI(Module):
    def __init__(self, di, dm, do, depth=3):
        self.embed = Linear(di, dm)
        self.blocks = [Block(dm) for _ in range(depth)]
        self.norm, self.head = RMSNorm(dm), Linear(dm, do)
    def f(self, x):
        if x.ndim == 2: x = x[:, None, :]
        x = self.embed.f(x)
        for b in self.blocks: x = b.f(x)
        return self.head.f(self.norm.f(x[:, -1]))
    def b(self, dy):
        dy = self.norm.b(self.head.b(dy))
        db = np.zeros((dy.shape[0], self.embed.x.shape[1], dy.shape[1]))
        db[:, -1] = dy
        for b in reversed(self.blocks): db = b.b(db)
        self.embed.b(db)

class AdamW:
    def __init__(self, p, lr=1e-3, b1=0.9, b2=0.999, wd=0.01):
        self.p, self.lr, self.b1, self.b2, self.wd = p, lr, b1, b2, wd
        self.m = [np.zeros_like(i.data) for i in p]
        self.v = [np.zeros_like(i.data) for i in p]
        self.t = 0
    def step(self):
        self.t += 1
        a = self.lr * (np.sqrt(1 - self.b2**self.t) / (1 - self.b1**self.t))
        for i, pt in enumerate(self.p):
            g = np.clip(pt.grad, -1.0, 1.0)
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g**2)
            pt.data -= a * (self.m[i] / (np.sqrt(self.v[i]) + 1e-8) + self.wd * pt.data)
            pt.grad.fill(0)

def main():
    N, D, C, BS, E = 1024, 784, 10, 64, 50
    X = np.random.randn(N, D).astype("f4")
    Y = np.random.randint(0, C, N)
    
    model = OMEGA_ASI(D, 128, C, depth=2)
    opt = AdamW(model.params(), lr=2e-3, wd=0.02)
    
    for epoch in range(E):
        idx = np.random.permutation(N)
        losses, accs = [], []
        for i in range(0, N, BS):
            xb, yb = X[idx[i:i+BS]], Y[idx[i:i+BS]]
            logits = model.f(xb)
            
            probs = (exp_l := np.exp(logits - logits.max(-1, keepdims=True))) / (exp_l.sum(-1, keepdims=True) + 1e-12)
            loss = -np.mean(np.log(probs[range(len(yb)), yb] + 1e-12))
            acc = np.mean(probs.argmax(-1) == yb)
            
            losses.append(loss)
            accs.append(acc)
            
            d_logits = probs.copy()
            d_logits[range(len(yb)), yb] -= 1
            model.b(d_logits / len(yb))
            opt.step()
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:03d} | Loss: {np.mean(losses):.4f} | Acc: {np.mean(accs):.4f}")

if __name__ == "__main__":
    main()
