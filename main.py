import numpy as np

def silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))

def dsilu(x):
    s = 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))
    return s * (1.0 + x * (1.0 - s))

class Linear:
    def __init__(self, i, o, name=""):
        self.W = np.random.randn(i, o).astype("f4") * np.sqrt(2.0 / i)
        self.b = np.zeros(o, "f4")
        self.dW, self.db = None, None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dy):
        x_flat = self.x.reshape(-1, self.x.shape[-1])
        dy_flat = dy.reshape(-1, dy.shape[-1])
        self.dW = x_flat.T @ dy_flat
        self.db = dy_flat.sum(axis=0)
        return (dy @ self.W.T).reshape(self.x.shape[:-1] + (self.W.shape[0],))

class RMSNorm:
    def __init__(self, d, eps=1e-6):
        self.g = np.ones(d, "f4")
        self.eps = eps

    def forward(self, x):
        self.x = x
        self.rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        self.xn = x / self.rms
        return self.g * self.xn

    def backward(self, dy):
        dn = dy * self.g
        self.dg = np.sum(dy * self.xn, axis=(0, 1)) if dy.ndim == 3 else np.sum(dy * self.xn, axis=0)
        dd = (dn - self.xn * np.mean(dn * self.xn, axis=-1, keepdims=True)) / self.rms
        return dd

class RoPE:
    def __init__(self, d, m=4096):
        f = 10000.0**-(np.arange(0, d, 2) / d)
        t = np.outer(np.arange(m), f)
        self.cos, self.sin = np.cos(t), np.sin(t)

    def apply(self, x, conj=False):
        s, d = x.shape[1], x.shape[-1]
        x1, x2 = x[..., :d//2], x[..., d//2:]
        c, sn = self.cos[:s, None, :], self.sin[:s, None, :]
        if conj: return np.concatenate([x1 * c + x2 * sn, x2 * c - x1 * sn], axis=-1)
        return np.concatenate([x1 * c - x2 * sn, x2 * c + x1 * sn], axis=-1)

class GQA:
    def __init__(self, d, h=8, g=2):
        self.d, self.h, self.g, self.hd = d, h, g, d // h
        self.wq = Linear(d, d)
        self.wk = Linear(d, (h // g) * self.hd)
        self.wv = Linear(d, (h // g) * self.hd)
        self.wo = Linear(d, d)
        self.rope = RoPE(self.hd)
        self.scale = self.hd**-0.5

    def forward(self, x):
        b, s, _ = x.shape
        q = self.wq.forward(x).reshape(b, s, self.h, self.hd)
        k = self.wk.forward(x).reshape(b, s, self.h // self.g, self.hd)
        v = self.wv.forward(x).reshape(b, s, self.h // self.g, self.hd)
        self.qr, self.kr, self.v_raw = self.rope.apply(q), self.rope.apply(k), v
        ke = np.repeat(self.kr, self.g, axis=2)
        ve = np.repeat(v, self.g, axis=2)
        attn = np.einsum("bshd,bthd->bsht", self.qr, ke) * self.scale
        attn -= np.max(attn, axis=-1, keepdims=True)
        self.p = np.exp(attn)
        self.p /= (np.sum(self.p, axis=-1, keepdims=True) + 1e-12)
        out = np.einsum("bsht,bthd->bshd", self.p, ve).reshape(b, s, -1)
        return self.wo.forward(out)

    def backward(self, dy):
        b, s, _ = dy.shape
        dy_o = self.wo.backward(dy).reshape(b, s, self.h, self.hd)
        ke = np.repeat(self.kr, self.g, axis=2)
        ve = np.repeat(self.v_raw, self.g, axis=2)
        dp = np.einsum("bshd,bthd->bsht", dy_o, ve)
        da = self.p * (dp - np.sum(self.p * dp, axis=-1, keepdims=True)) * self.scale
        dqr = np.einsum("bsht,bthd->bshd", da, ke)
        dke = np.einsum("bsht,bshd->bthd", da, self.qr)
        dve = np.einsum("bsht,bshd->bthd", self.p, dy_o)
        dq = self.rope.apply(dqr, conj=True).reshape(b, s, -1)
        dk = self.rope.apply(dke.reshape(b, s, self.h // self.g, self.g, self.hd).sum(axis=3), conj=True).reshape(b, s, -1)
        dv = dve.reshape(b, s, self.h // self.g, self.g, self.hd).sum(axis=3).reshape(b, s, -1)
        return self.wq.backward(dq) + self.wk.backward(dk) + self.wv.backward(dv)

class SwiGLU:
    def __init__(self, d, h):
        self.w1 = Linear(d, h)
        self.w2 = Linear(h, d)
        self.w3 = Linear(d, h)

    def forward(self, x):
        self.x1, self.x3 = self.w1.forward(x), self.w3.forward(x)
        self.sig = silu(self.x1)
        return self.w2.forward(self.sig * self.x3)

    def backward(self, dy):
        da = self.w2.backward(dy)
        dx3 = da * self.sig
        dx1 = da * self.x3 * dsilu(self.x1)
        return self.w1.backward(dx1) + self.w3.backward(dx3)

class MoE:
    def __init__(self, d, n=4, e=2):
        self.n = n
        self.gate = Linear(d, n)
        self.experts = [SwiGLU(d, d * e) for _ in range(n)]

    def forward(self, x):
        self.x = x
        g = self.gate.forward(x)
        self.p = np.exp(g - np.max(g, axis=-1, keepdims=True))
        self.p /= np.sum(self.p, axis=-1, keepdims=True)
        self.ex_out = [exp.forward(x) for exp in self.experts]
        return sum(self.p[..., i:i+1] * self.ex_out[i] for i in range(self.n))

    def backward(self, dy):
        dx = np.zeros_like(self.x)
        dp = np.zeros_like(self.p)
        for i in range(self.n):
            dp[..., i] = np.sum(dy * self.ex_out[i], axis=-1)
            dx += self.experts[i].backward(dy * self.p[..., i:i+1])
        dg = self.p * (dp - np.sum(self.p * dp, axis=-1, keepdims=True))
        return dx + self.gate.backward(dg)

class SovereignLogic:
    def __init__(self, d):
        self.gemini_path = GQA(d)
        self.groq_path = Linear(d, d)
        self.alpha = np.array([0.5], dtype="f4")
        self.d_alpha = 0

    def forward(self, x):
        self.ga = self.gemini_path.forward(x)
        self.gr = self.groq_path.forward(x)
        return self.alpha * self.ga + (1.0 - self.alpha) * self.gr

    def backward(self, dy):
        self.d_alpha = np.sum(dy * (self.ga - self.gr))
        dga = self.gemini_path.backward(dy * self.alpha)
        dgr = self.groq_path.backward(dy * (1.0 - self.alpha))
        return dga + dgr

class Block:
    def __init__(self, d):
        self.n1 = RMSNorm(d)
        self.sl = SovereignLogic(d)
        self.n2 = RMSNorm(d)
        self.moe = MoE(d)

    def forward(self, x):
        x = x + self.sl.forward(self.n1.forward(x))
        x = x + self.moe.forward(self.n2.forward(x))
        return x

    def backward(self, dy):
        dy_moe = self.moe.backward(dy)
        dy = dy + self.n2.backward(dy_moe)
        dy_sl = self.sl.backward(dy)
        dy = dy + self.n1.backward(dy_sl)
        return dy

class SovereignASI:
    def __init__(self, i, h, o, depth=2):
        self.emb = Linear(i, h)
        self.blocks = [Block(h) for _ in range(depth)]
        self.norm = RMSNorm(h)
        self.head = Linear(h, o)

    def forward(self, x):
        if x.ndim == 2: x = x[:, None, :]
        x = self.emb.forward(x)
        for b in self.blocks: x = b.forward(x)
        self.last_x = self.norm.forward(x[:, -1, :])
        return self.head.forward(self.last_x)

    def backward(self, dy):
        dy = self.head.backward(dy)
        dy = self.norm.backward(dy)
        dy_seq = np.zeros((dy.shape[0], 1, dy.shape[1]), dtype="f4")
        dy_seq[:, -1, :] = dy
        for b in reversed(self.blocks): dy_seq = b.backward(dy_seq)
        self.emb.backward(dy_seq)

    def get_params(self):
        params = []
        def collect(obj):
            if isinstance(obj, (Linear, RMSNorm)): params.append(obj)
            elif isinstance(obj, list): [collect(i) for i in obj]
            elif hasattr(obj, "__dict__"):
                for k, v in obj.__dict__.items():
                    if k != "rope": collect(v)
        collect(self)
        return list(set(params))

class AdamW:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, wd=0.01):
        self.params = params
        self.lr, self.b1, self.b2, self.wd = lr, b1, b2, wd
        self.t = 0
        self.m = {id(p): [np.zeros_like(getattr(p, a)) for a in (["W", "b"] if hasattr(p, "W") else ["g"])] for p in params}
        self.v = {id(p): [np.zeros_like(getattr(p, a)) for a in (["W", "b"] if hasattr(p, "W") else ["g"])] for p in params}

    def step(self):
        self.t += 1
        lr_t = self.lr * min(1.0, self.t / 100)
        for p in self.params:
            attrs = ["W", "b"] if hasattr(p, "W") else ["g"]
            for i, a in enumerate(attrs):
                grad = getattr(p, "d" + a if a != "g" else "dg")
                idx = id(p)
                self.m[idx][i] = self.b1 * self.m[idx][i] + (1.0 - self.b1) * grad
                self.v[idx][i] = self.b2 * self.v[idx][i] + (1.0 - self.b2) * (grad**2)
                mt = self.m[idx][i] / (1.0 - self.b1**self.t)
                vt = self.v[idx][i] / (1.0 - self.b2**self.t)
                val = getattr(p, a)
                val -= lr_t * (mt / (np.sqrt(vt) + 1e-8) + self.wd * val)
                setattr(p, a, val)
            if hasattr(p, "alpha"): p.alpha -= lr_t * p.d_alpha

def train():
    N, D, C, BS, E = 1024, 784, 10, 32, 50
    X = np.random.randn(N, D).astype("f4")
    Y = np.random.randint(0, C, N)
    model = SovereignASI(D, 128, C, depth=2)
    params = model.get_params()
    opt = AdamW(params, lr=2e-3)

    for e in range(E):
        indices = np.random.permutation(N)
        total_loss, total_acc = 0, 0
        for i in range(0, N, BS):
            batch_idx = indices[i:i+BS]
            xb, yb = X[batch_idx], Y[batch_idx]
            
            logits = model.forward(xb)
            probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs /= np.sum(probs, axis=-1, keepdims=True)
            
            loss = -np.mean(np.log(probs[np.arange(len(yb)), yb] + 1e-12))
            total_loss += loss * len(yb)
            total_acc += np.sum(np.argmax(probs, axis=-1) == yb)
            
            dy = probs.copy()
            dy[np.arange(len(yb)), yb] -= 1
            model.backward(dy / len(yb))
            
            gnorm = np.sqrt(sum(np.sum(getattr(p, "d"+a if a!="g" else "dg")**2) for p in params for a in (["W","b"] if hasattr(p,"W") else ["g"])))
            if gnorm > 1.0:
                for p in params:
                    for a in (["W","b"] if hasattr(p,"W") else ["g"]):
                        k = "d"+a if a!="g" else "dg"
                        setattr(p, k, getattr(p, k) / gnorm)
            opt.step()
            
        if (e + 1) % 5 == 0:
            print(f"Epoch {e+1:02d} | Loss: {total_loss/N:.4f} | Acc: {total_acc/N:.4f}")

if __name__ == "__main__":
    train()
