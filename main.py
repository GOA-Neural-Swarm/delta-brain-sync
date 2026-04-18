import numpy as np

class Tensor:
    def __init__(self, data, name=""):
        self.data = np.ascontiguousarray(data).astype("f4")
        self.grad = np.zeros_like(self.data)
        self.name = name

class Module:
    def params(self):
        p = []
        for v in self.__dict__.values():
            if isinstance(v, Tensor): p.append(v)
            elif isinstance(v, Module): p.extend(v.params())
            elif isinstance(v, list):
                for i in v:
                    if isinstance(i, Module): p.extend(i.params())
                    elif isinstance(i, Tensor): p.append(i)
        return p

class Linear(Module):
    def __init__(self, i, o, bias=True):
        self.w = Tensor(np.random.randn(i, o) * np.sqrt(2 / i))
        self.b = Tensor(np.zeros(o)) if bias else None

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
        self.g, self.e = Tensor(np.ones(d)), e

    def f(self, x):
        self.x = x
        self.ms = np.mean(x**2, -1, keepdims=True)
        self.r = 1.0 / np.sqrt(self.ms + self.e)
        return self.g.data * (x * self.r)

    def b(self, dy):
        x_norm = self.x * self.r
        self.g.grad += np.sum(dy * x_norm, axis=tuple(range(dy.ndim - 1)))
        dx_norm = dy * self.g.data
        return self.r * (dx_norm - x_norm * np.mean(dx_norm * x_norm, -1, keepdims=True))

class SwiGLU(Module):
    def f(self, x):
        x1, x2 = np.split(x, 2, axis=-1)
        self.x1, self.x2 = x1, x2
        self.sig = 1 / (1 + np.exp(-np.clip(x1, -12, 12)))
        self.swish = x1 * self.sig
        return self.swish * x2

    def b(self, dy):
        ds = dy * self.x2
        dx2 = dy * self.swish
        dx1 = ds * (self.sig * (1 + self.x1 * (1 - self.sig)))
        return np.concatenate([dx1, dx2], axis=-1)

class RotaryAttention(Module):
    def __init__(self, d, h=8):
        self.d, self.h, self.hd = d, h, d // h
        self.wq = Linear(d, d, False)
        self.wk = Linear(d, d, False)
        self.wv = Linear(d, d, False)
        self.wo = Linear(d, d, False)
        inv_freq = 1.0 / (10000 ** (np.arange(0, self.hd, 2) / self.hd))
        t = np.arange(2048)
        freqs = np.outer(t, inv_freq)
        self.cos, self.sin = np.cos(freqs), np.sin(freqs)

    def _apply_rope(self, x):
        s = x.shape[1]
        c, sn = self.cos[:s, None, :], self.sin[:s, None, :]
        r, i = x[..., ::2], x[..., 1::2]
        o = np.empty_like(x)
        o[..., ::2], o[..., 1::2] = r * c - i * sn, r * sn + i * c
        return o

    def _rope_grad(self, dy):
        s = dy.shape[1]
        c, sn = self.cos[:s, None, :], self.sin[:s, None, :]
        dr, di = dy[..., ::2], dy[..., 1::2]
        o = np.empty_like(dy)
        o[..., ::2], o[..., 1::2] = dr * c + di * sn, di * c - dr * sn
        return o

    def f(self, x):
        b, s, _ = x.shape
        q = self.wq.f(x).reshape(b, s, self.h, self.hd)
        k = self.wk.f(x).reshape(b, s, self.h, self.hd)
        v = self.wv.f(x).reshape(b, s, self.h, self.hd)
        self.q_rope, self.k_rope = self._apply_rope(q), self._apply_rope(k)
        self.v_cached = v
        score = np.einsum("bshd,bthd->bsht", self.q_rope, self.k_rope) * (self.hd**-0.5)
        self.p = (e := np.exp(score - score.max(-1, keepdims=True))) / (e.sum(-1, keepdims=True) + 1e-12)
        ctx = np.einsum("bsht,bthd->bshd", self.p, v).reshape(b, s, -1)
        return self.wo.f(ctx)

    def b(self, dy):
        b, s = dy.shape[:2]
        do = self.wo.b(dy).reshape(b, s, self.h, self.hd)
        dp = np.einsum("bshd,bthd->bsht", do, self.v_cached)
        ds = self.p * (dp - (self.p * dp).sum(-1, keepdims=True)) * (self.hd**-0.5)
        dq_rope = np.einsum("bsht,bthd->bshd", ds, self.k_rope)
        dk_rope = np.einsum("bsht,bshd->bthd", ds, self.q_rope)
        dv = np.einsum("bsht,bshd->bthd", self.p, do)
        dq = self._rope_grad(dq_rope)
        dk = self._rope_grad(dk_rope)
        return self.wq.b(dq.reshape(b, s, -1)) + self.wk.b(dk.reshape(b, s, -1)) + self.wv.b(dv.reshape(b, s, -1))

class MoE(Module):
    def __init__(self, d, n=4, k=1):
        self.d, self.n, self.k = d, n, k
        self.gate = Linear(d, n, False)
        self.experts = [[Linear(d, d*2, False), SwiGLU(), Linear(d*2, d, False)] for _ in range(n)]

    def f(self, x):
        self.sh = x.shape
        xf = x.reshape(-1, self.d)
        logits = self.gate.f(xf)
        p = (e := np.exp(logits - logits.max(-1, keepdims=True))) / (e.sum(-1, keepdims=True) + 1e-12)
        self.idx = np.argsort(p, -1)[:, -self.k:]
        self.w = np.take_along_axis(p, self.idx, -1)
        self.w /= self.w.sum(-1, keepdims=True) + 1e-12
        out, self.cache = np.zeros_like(xf), []
        for i in range(self.n):
            mask = np.any(self.idx == i, axis=-1)
            if not np.any(mask):
                self.cache.append(None)
                continue
            pos = np.where(self.idx[mask] == i)[1]
            h1 = self.experts[i][0].f(xf[mask])
            h2 = self.experts[i][1].f(h1)
            h3 = self.experts[i][2].f(h2)
            out[mask] += h3 * self.w[mask, pos][:, None]
            self.cache.append((mask, pos, h1, h2, h3))
        return out.reshape(self.sh)

    def b(self, dy):
        dyf = dy.reshape(-1, self.d)
        dx, dg = np.zeros_like(dyf), np.zeros((dyf.shape[0], self.n))
        for i in range(self.n):
            if self.cache[i] is None: continue
            mask, pos, h1, h2, h3 = self.cache[i]
            dg[mask, i] = (dyf[mask] * h3).sum(-1)
            dh3 = dyf[mask] * self.w[mask, pos][:, None]
            dh2 = self.experts[i][2].b(dh3)
            dh1 = self.experts[i][1].b(dh2)
            dx[mask] += self.experts[i][0].b(dh1)
        return (dx + self.gate.b(dg - dg.mean(-1, keepdims=True))).reshape(self.sh)

class RedundantConsensusBlock(Module):
    def __init__(self, d):
        self.norm_g = RMSNorm(d)
        self.gemini_stream = RotaryAttention(d)
        self.norm_q = RMSNorm(d)
        self.groq_stream = MoE(d)
        self.gate = Linear(d, 2, False)

    def f(self, x):
        self.x = x
        self.o_gemini = self.gemini_stream.f(self.norm_g.f(x))
        self.o_groq = self.groq_stream.f(self.norm_q.f(x))
        g_logits = self.gate.f(np.mean(x, axis=1))
        self.p = (e := np.exp(g_logits - g_logits.max(-1, keepdims=True))) / (e.sum(-1, keepdims=True) + 1e-12)
        return x + self.p[:, 0:1, None] * self.o_gemini + self.p[:, 1:2, None] * self.o_groq

    def b(self, dy):
        dg_logits = np.zeros_like(self.p)
        dg_logits[:, 0] = np.sum(dy * self.o_gemini, axis=(1, 2))
        dg_logits[:, 1] = np.sum(dy * self.o_groq, axis=(1, 2))
        dx_gate = self.gate.b(dg_logits - dg_logits.mean(-1, keepdims=True))
        
        dg_stream = dy * self.p[:, 0:1, None]
        dq_stream = dy * self.p[:, 1:2, None]
        
        dx = dy + self.norm_g.b(self.gemini_stream.b(dg_stream)) + self.norm_q.b(self.groq_stream.b(dq_stream))
        dx += (dx_gate[:, None, :] / self.x.shape[1])
        return dx

class OMEGA_ASI(Module):
    def __init__(self, di, dm, do, depth=2):
        self.embed = Linear(di, dm)
        self.blocks = [RedundantConsensusBlock(dm) for _ in range(depth)]
        self.final_norm = RMSNorm(dm)
        self.head = Linear(dm, do)

    def f(self, x):
        x = self.embed.f(x[:, None] if x.ndim == 2 else x)
        for b in self.blocks: x = b.f(x)
        return self.head.f(self.final_norm.f(x[:, -1]))

    def b(self, dy):
        dy = self.final_norm.b(self.head.b(dy))
        db = np.zeros((dy.shape[0], self.embed.x.shape[1], dy.shape[1]))
        db[:, -1] = dy
        for b in reversed(self.blocks): db = b.b(db)
        return self.embed.b(db)

class AdamW:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, wd=0.01):
        self.p, self.lr, self.b1, self.b2, self.wd, self.t = params, lr, b1, b2, wd, 0
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]

    def step(self):
        self.t += 1
        a = self.lr * (np.sqrt(1 - self.b2**self.t) / (1 - self.b1**self.t))
        for i, p in enumerate(self.p):
            g = np.clip(p.grad, -5.0, 5.0)
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g**2)
            p.data -= a * (self.m[i] / (np.sqrt(self.v[i]) + 1e-8) + self.wd * p.data)
            p.grad.fill(0)

if __name__ == "__main__":
    N, D, C, BS, E = 2048, 784, 10, 64, 100
    X = np.random.randn(N, D).astype("f4")
    Y = np.random.randint(0, C, N)
    
    model = OMEGA_ASI(D, 128, C, depth=2)
    optimizer = AdamW(model.params(), lr=1e-3, wd=0.01)
    
    for e in range(E):
        idx = np.random.permutation(N)
        L, A = [], []
        for i in range(0, N, BS):
            xb, yb = X[idx[i:i+BS]], Y[idx[i:i+BS]]
            logits = model.f(xb)
            
            ex = np.exp(logits - np.max(logits, -1, keepdims=True))
            probs = ex / (ex.sum(-1, keepdims=True) + 1e-12)
            
            loss = -np.mean(np.log(probs[range(len(yb)), yb] + 1e-12))
            acc = np.mean(probs.argmax(-1) == yb)
            
            L.append(loss); A.append(acc)
            
            d_logits = probs.copy()
            d_logits[range(len(yb)), yb] -= 1
            model.b(d_logits / len(yb))
            optimizer.step()
            
        if (e + 1) % 5 == 0:
            print(f"STEP {e+1:03} | LOSS: {np.mean(L):.4f} | ACC: {np.mean(A):.4f}")
