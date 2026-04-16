import numpy as np

class Tensor:
    def __init__(self, data, name=""):
        self.data = data.astype("f4")
        self.grad = np.zeros_like(self.data)

class Linear:
    def __init__(self, i, o, name=""):
        scale = np.sqrt(2.0 / i)
        self.w = Tensor(np.random.normal(0, scale, (i, o)), f"{name}_w")
        self.b = Tensor(np.zeros(o), f"{name}_b")

    def forward(self, x):
        self.x = x
        return x @ self.w.data + self.b.data

    def backward(self, dy):
        self.w.grad += self.x.reshape(-1, self.x.shape[-1]).T @ dy.reshape(-1, dy.shape[-1])
        self.b.grad += dy.reshape(-1, dy.shape[-1]).sum(0)
        return dy @ self.w.data.T

class RMSNorm:
    def __init__(self, d, e=1e-6):
        self.g = Tensor(np.ones(d))
        self.e = e

    def forward(self, x):
        self.x = x
        self.v = np.mean(x**2, -1, keepdims=True)
        self.inv = 1. / np.sqrt(self.v + self.e)
        self.nx = x * self.inv
        return self.g.data * self.nx

    def backward(self, dy):
        dg = dy * self.g.data
        self.g.grad += np.sum(dy * self.nx, axis=tuple(range(dy.ndim - 1)))
        return (dg - self.nx * np.mean(dg * self.nx, -1, keepdims=True)) * self.inv

class GQA:
    def __init__(self, d, h=8, g=2):
        self.d, self.h, self.g, self.hd = d, h, g, d // h
        self.wq = Linear(d, d)
        self.wk = Linear(d, (h // g) * self.hd)
        self.wv = Linear(d, (h // g) * self.hd)
        self.wo = Linear(d, d)
        self.scale = self.hd**-0.5

    def _rope(self, t, inv=False):
        b, s, h, d = t.shape
        p = np.arange(s)[:, None]
        f = 10000**-(np.arange(0, d, 2) / d)
        a = p * f
        cos, sin = np.cos(a), np.sin(a)
        if inv: sin = -sin
        r, i = t[..., ::2], t[..., 1::2]
        out = np.empty_like(t)
        out[..., ::2] = r * cos[:, None, :] - i * sin[:, None, :]
        out[..., 1::2] = r * sin[:, None, :] + i * cos[:, None, :]
        return out

    def forward(self, x):
        b, s, _ = x.shape
        q = self.wq.forward(x).reshape(b, s, self.h, self.hd)
        k = self.wk.forward(x).reshape(b, s, self.h // self.g, self.hd)
        v = self.wv.forward(x).reshape(b, s, self.h // self.g, self.hd)
        self.q_r, self.k_r, self.v_r = self._rope(q), self._rope(k), v
        k_rep = np.repeat(self.k_r, self.g, 2)
        v_rep = np.repeat(v, self.g, 2)
        attn = np.einsum("bshd,bthd->bsht", self.q_r, k_rep) * self.scale
        self.p = np.exp(attn - np.max(attn, -1, keepdims=True))
        self.p /= (self.p.sum(-1, keepdims=True) + 1e-12)
        out = np.einsum("bsht,bthd->bshd", self.p, v_rep).reshape(b, s, -1)
        return self.wo.forward(out)

    def backward(self, dy):
        b, s = dy.shape[:2]
        dy_wo = self.wo.backward(dy).reshape(b, s, self.h, self.hd)
        k_rep = np.repeat(self.k_r, self.g, 2)
        v_rep = np.repeat(self.v_r, self.g, 2)
        dp = np.einsum("bshd,bthd->bsht", dy_wo, v_rep)
        da = self.p * (dp - np.sum(self.p * dp, -1, keepdims=True)) * self.scale
        dq_r = np.einsum("bsht,bthd->bshd", da, k_rep)
        dk_r_full = np.einsum("bsht,bshd->bthd", da, self.q_r)
        dk_r = dk_r_full.reshape(b, s, self.h // self.g, self.g, self.hd).sum(3)
        dv_full = np.einsum("bsht,bshd->bthd", self.p, dy_wo)
        dv = dv_full.reshape(b, s, self.h // self.g, self.g, self.hd).sum(3)
        dq = self.wq.backward(self._rope(dq_r, True).reshape(b, s, -1))
        dk = self.wk.backward(self._rope(dk_r, True).reshape(b, s, -1))
        dv = self.wv.backward(dv.reshape(b, s, -1))
        return dq + dk + dv

class MoE:
    def __init__(self, d, n=4, k=2):
        self.d, self.n, self.k = d, n, k
        self.gate = Linear(d, n)
        self.experts_w1 = [Linear(d, d * 2) for _ in range(n)]
        self.experts_w2 = [Linear(d * 2, d) for _ in range(n)]

    def _swiglu(self, x):
        x, g = np.split(x, 2, -1)
        sig = 1. / (1. + np.exp(-np.clip(g, -15, 15)))
        return x * (g * sig), (x, g, sig)

    def _swiglu_back(self, dy, cache):
        x, g, sig = cache
        dx = dy * (g * sig)
        dg = dy * x * sig * (1. + g * (1. - sig))
        return np.concatenate([dx, dg], -1)

    def forward(self, x):
        self.sh = x.shape
        xf = x.reshape(-1, self.d)
        logits = self.gate.forward(xf)
        p = np.exp(logits - np.max(logits, -1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        self.idx = np.argsort(p, -1)[:, -self.k:]
        self.w = np.take_along_axis(p, self.idx, -1)
        self.w /= (self.w.sum(-1, keepdims=True) + 1e-12)
        out = np.zeros_like(xf)
        self.cache = []
        for i in range(self.n):
            mask = np.any(self.idx == i, axis=-1)
            if not np.any(mask):
                self.cache.append(None)
                continue
            pos = np.where(self.idx[mask] == i)[1]
            wi = self.w[mask, pos][:, None]
            h1 = self.experts_w1[i].forward(xf[mask])
            act, c = self._swiglu(h1)
            h2 = self.experts_w2[i].forward(act)
            out[mask] += h2 * wi
            self.cache.append((mask, pos, act, c))
        return out.reshape(self.sh)

    def backward(self, dy):
        dyf = dy.reshape(-1, self.d)
        dx, dg = np.zeros((dyf.shape[0], self.d)), np.zeros((dyf.shape[0], self.n))
        for i in range(self.n):
            if self.cache[i] is None: continue
            mask, pos, act, c = self.cache[i]
            wi = self.w[mask, pos][:, None]
            dyi = dyf[mask] * wi
            dg[mask, i] = np.sum(dyf[mask] * self.experts_w2[i].forward(act), -1)
            dact = self.experts_w2[i].backward(dyi)
            dh1 = self._swiglu_back(dact, c)
            dx[mask] += self.experts_w1[i].backward(dh1)
        return (dx + self.gate.backward(dg - np.mean(dg, -1, keepdims=True))).reshape(self.sh)

class RedundantCompute:
    def __init__(self, d):
        self.gemini_path = MoE(d, n=4, k=2)
        self.groq_path = GQA(d, h=4, g=2)
        self.alpha = Tensor(np.array([0.5]))

    def forward(self, x):
        self.x = x
        self.out_gemini = self.gemini_path.forward(x)
        self.out_groq = self.groq_path.forward(x)
        return self.alpha.data * self.out_gemini + (1 - self.alpha.data) * self.out_groq

    def backward(self, dy):
        d_gemini = self.gemini_path.backward(dy * self.alpha.data)
        d_groq = self.groq_path.backward(dy * (1 - self.alpha.data))
        self.alpha.grad += np.sum(dy * (self.out_gemini - self.out_groq))
        return d_gemini + d_groq

class SovereignBlock:
    def __init__(self, d):
        self.n1 = RMSNorm(d)
        self.at = GQA(d)
        self.n2 = RMSNorm(d)
        self.rc = RedundantCompute(d)

    def forward(self, x):
        x = x + self.at.forward(self.n1.forward(x))
        x = x + self.rc.forward(self.n2.forward(x))
        return x

    def backward(self, dy):
        dy_rc = self.rc.backward(self.n2.backward(dy))
        dy = dy + dy_rc
        dy_at = self.at.backward(self.n1.backward(dy))
        return dy + dy_at

class OMEGA_ASI:
    def __init__(self, di, dm, do, depth=2):
        self.embed = Linear(di, dm)
        self.blocks = [SovereignBlock(dm) for _ in range(depth)]
        self.fn = RMSNorm(dm)
        self.head = Linear(dm, do)
        self.params = self._collect_params()

    def _collect_params(self):
        p = []
        def _walk(obj):
            if isinstance(obj, Tensor): p.append(obj)
            elif hasattr(obj, 'w'): p.extend([obj.w, obj.b])
            elif hasattr(obj, 'g'): p.append(obj.g)
            elif isinstance(obj, list): [_walk(i) for i in obj]
            elif hasattr(obj, '__dict__'): [_walk(v) for v in obj.__dict__.values()]
        _walk(self)
        return p

    def forward(self, x):
        x = self.embed.forward(x[:, None] if x.ndim == 2 else x)
        for b in self.blocks: x = b.forward(x)
        return self.head.forward(self.fn.forward(x[:, -1]))

    def backward(self, dy):
        dy = self.fn.backward(self.head.backward(dy))
        db = np.zeros((dy.shape[0], self.embed.x.shape[1], dy.shape[1]))
        db[:, -1] = dy
        for b in reversed(self.blocks): db = b.backward(db)
        self.embed.backward(db)

class AdamW:
    def __init__(self, p, lr=1e-3, wd=0.01, b1=0.9, b2=0.999):
        self.p, self.lr, self.wd, self.b1, self.b2 = p, lr, wd, b1, b2
        self.m = [np.zeros_like(i.data) for i in p]
        self.v = [np.zeros_like(i.data) for i in p]
        self.t = 0

    def step(self):
        self.t += 1
        lr_t = self.lr * (np.sqrt(1 - self.b2**self.t) / (1 - self.b1**self.t))
        for i, pt in enumerate(self.p):
            g = np.clip(pt.grad, -1, 1)
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g**2)
            pt.data -= lr_t * (self.m[i] / (np.sqrt(self.v[i]) + 1e-8) + self.wd * pt.data)
            pt.grad.fill(0)

def train():
    N, D, C, BS, E = 1024, 784, 10, 64, 50
    X = np.random.randn(N, D).astype("f4")
    Y = np.random.randint(0, C, N)
    model = OMEGA_ASI(D, 128, C, depth=2)
    optimizer = AdamW(model.params, lr=2e-3, wd=0.01)

    for epoch in range(E):
        indices = np.random.permutation(N)
        losses, accs = [], []
        for i in range(0, N, BS):
            batch_idx = indices[i:i+BS]
            xb, yb = X[batch_idx], Y[batch_idx]
            
            logits = model.forward(xb)
            probs = np.exp(logits - np.max(logits, -1, keepdims=True))
            probs /= probs.sum(-1, keepdims=True)
            
            loss = -np.mean(np.log(probs[np.arange(len(yb)), yb] + 1e-12))
            acc = np.mean(np.argmax(probs, -1) == yb)
            losses.append(loss)
            accs.append(acc)
            
            d_logits = probs.copy()
            d_logits[np.arange(len(yb)), yb] -= 1
            model.backward(d_logits / len(yb))
            optimizer.step()
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:03d} | Loss: {np.mean(losses):.4f} | Acc: {np.mean(accs):.4f}")

if __name__ == "__main__":
    train()
