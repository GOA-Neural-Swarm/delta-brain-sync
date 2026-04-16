import numpy as np

def swiglu(x):
    x, gate = np.split(x, 2, axis=-1)
    sig = 1.0 / (1.0 + np.exp(-np.clip(gate, -15, 15)))
    swi = gate * sig
    return x * swi, (x, gate, sig, swi)

def swiglu_back(dy, cache):
    x, gate, sig, swi = cache
    dx = dy * swi
    dgate = dy * x * sig * (1.0 + gate * (1.0 - sig))
    return np.concatenate([dx, dgate], axis=-1)

class Linear:
    def __init__(self, i, o, std=0.02):
        self.W = np.random.normal(0, std, (i, o)).astype("f4")
        self.b = np.zeros(o, "f4")
        self.dW, self.db = None, None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dy):
        x_flat = self.x.reshape(-1, self.x.shape[-1])
        dy_flat = dy.reshape(-1, dy.shape[-1])
        self.dW = x_flat.T @ dy_flat
        self.db = dy_flat.sum(0)
        return (dy @ self.W.T).reshape(self.x.shape)

class RMSNorm:
    def __init__(self, d, e=1e-6):
        self.g, self.e = np.ones(d, "f4"), e

    def forward(self, x):
        self.x = x
        self.r = np.sqrt(np.mean(x**2, -1, keepdims=True) + self.e)
        self.xn = x / self.r
        return self.g * self.xn

    def backward(self, dy):
        dn = dy * self.g
        self.dg = np.sum(dy * self.xn, axis=tuple(range(dy.ndim - 1)))
        return (dn - self.xn * np.mean(dn * self.xn, -1, keepdims=True)) / self.r

class RoPE:
    def __init__(self, d, m=2048):
        f = 10000.0 ** -(np.arange(0, d, 2) / d)
        t = np.outer(np.arange(m), f)
        self.c, self.s = np.cos(t), np.sin(t)

    def apply(self, x, conj=False):
        s, d = x.shape[1], x.shape[-1]
        x1, x2 = x[..., :d//2], x[..., d//2:]
        c, sn = self.c[:s, None, :], self.s[:s, None, :]
        return np.concatenate([x1 * c + (x2 * sn if conj else -x2 * sn), 
                               x2 * c + (-x1 * sn if conj else x1 * sn)], -1)

class GQA:
    def __init__(self, d, h=8, g=2):
        self.d, self.h, self.g, self.hd = d, h, g, d // h
        self.wq = Linear(d, d)
        self.wk = Linear(d, d // g)
        self.wv = Linear(d, d // g)
        self.wo = Linear(d, d)
        self.rope = RoPE(self.hd)
        self.scale = self.hd ** -0.5

    def forward(self, x):
        b, s, _ = x.shape
        q = self.wq.forward(x).reshape(b, s, self.h, self.hd)
        k = self.wk.forward(x).reshape(b, s, self.h // self.g, self.hd)
        v = self.wv.forward(x).reshape(b, s, self.h // self.g, self.hd)
        self.qr, self.kr, self.v_raw = self.rope.apply(q), self.rope.apply(k), v
        ke = np.repeat(self.kr, self.g, 2)
        ve = np.repeat(v, self.g, 2)
        at = np.einsum("bshd,bthd->bsht", self.qr, ke) * self.scale
        at -= np.max(at, -1, keepdims=True)
        self.p = np.exp(at)
        self.p /= np.sum(self.p, -1, keepdims=True) + 1e-12
        out = np.einsum("bsht,bthd->bshd", self.p, ve).reshape(b, s, -1)
        return self.wo.forward(out)

    def backward(self, dy):
        b, s = dy.shape[:2]
        dyo = self.wo.backward(dy).reshape(b, s, self.h, self.hd)
        ke = np.repeat(self.kr, self.g, 2)
        ve = np.repeat(self.v_raw, self.g, 2)
        dp = np.einsum("bshd,bthd->bsht", dyo, ve)
        da = self.p * (dp - np.sum(self.p * dp, -1, keepdims=True)) * self.scale
        dqr = np.einsum("bsht,bthd->bshd", da, ke)
        dkr = np.einsum("bsht,bshd->bthd", da, self.qr).reshape(b, s, self.h // self.g, self.g, self.hd).sum(3)
        dv = np.einsum("bsht,bshd->bthd", self.p, dyo).reshape(b, s, self.h // self.g, self.g, self.hd).sum(3)
        dq = self.rope.apply(dqr, True).reshape(b, s, -1)
        dk = self.rope.apply(dkr, True).reshape(b, s, -1)
        return self.wq.backward(dq) + self.wk.backward(dk) + self.wv.backward(dv.reshape(b, s, -1))

class MoE:
    def __init__(self, d, n=4, k=2):
        self.d, self.n, self.k = d, n, k
        self.gate = Linear(d, n)
        self.experts = [[Linear(d, d * 2), Linear(d, d)] for _ in range(n)]

    def forward(self, x):
        self.sh = x.shape
        xf = x.reshape(-1, self.d)
        lg = self.gate.forward(xf)
        p = np.exp(lg - np.max(lg, -1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        
        # Groq-style deterministic top-k logic
        self.idx = np.argsort(p, -1)[:, -self.k:]
        self.gw = np.take_along_axis(p, self.idx, -1)
        self.gw /= self.gw.sum(-1, keepdims=True) + 1e-12
        
        out = np.zeros_like(xf)
        self.caches = []
        for i in range(self.n):
            mask = np.any(self.idx == i, axis=-1)
            if not np.any(mask):
                self.caches.append(None)
                continue
            ix = xf[mask]
            f1 = self.experts[i][0].forward(ix)
            act, c = swiglu(f1)
            f2 = self.experts[i][1].forward(act)
            
            # Gemini-style redundant weighting
            w_idx = np.where(self.idx == i)
            expert_weights = self.gw[w_idx[0], w_idx[1]][:, None]
            out[mask] += f2 * expert_weights
            self.caches.append((mask, c, act, f2, expert_weights))
        return out.reshape(self.sh)

    def backward(self, dy):
        df = dy.reshape(-1, self.d)
        dx = np.zeros((df.shape[0], self.d), "f4")
        dg = np.zeros((df.shape[0], self.n), "f4")
        
        for i in range(self.n):
            if self.caches[i] is None: continue
            mask, c, act, f2, ew = self.caches[i]
            
            # Gradient of expert output w.r.t weights
            dg[mask, i] = np.sum(df[mask] * f2, -1)
            
            # Backprop through expert
            df2 = df[mask] * ew
            dact = self.experts[i][1].backward(df2)
            df1 = swiglu_back(dact, c)
            dx[mask] += self.experts[i][0].backward(df1)
            
        # Backprop through gate with load balancing
        d_gate = self.gate.backward(dg - np.mean(dg, -1, keepdims=True))
        return (dx + d_gate).reshape(self.sh)

class SovereignBlock:
    def __init__(self, d):
        self.rms1 = RMSNorm(d)
        self.attn = GQA(d)
        self.rms2 = RMSNorm(d)
        self.moe = MoE(d)

    def forward(self, x):
        x = x + self.attn.forward(self.rms1.forward(x))
        x = x + self.moe.forward(self.rms2.forward(x))
        return x

    def backward(self, dy):
        dy_moe = self.moe.backward(self.rms2.backward(dy))
        dy = dy + dy_moe
        dy_attn = self.attn.backward(self.rms1.backward(dy))
        return dy + dy_attn

class OMEGA_ASI:
    def __init__(self, i, h, o, depth=2):
        self.emb = Linear(i, h)
        self.blocks = [SovereignBlock(h) for _ in range(depth)]
        self.norm = RMSNorm(h)
        self.head = Linear(h, o)

    def forward(self, x):
        if x.ndim == 2: x = x[:, None, :]
        x = self.emb.forward(x)
        for b in self.blocks: x = b.forward(x)
        self.latent = self.norm.forward(x[:, -1, :])
        return self.head.forward(self.latent)

    def backward(self, dy):
        dy = self.norm.backward(self.head.backward(dy))
        db = np.zeros((dy.shape[0], self.emb.x.shape[1], dy.shape[1]), "f4")
        db[:, -1, :] = dy
        for b in reversed(self.blocks): db = b.backward(db)
        self.emb.backward(db)

class AdamW:
    def __init__(self, model, lr=1e-3, wd=0.01, b1=0.9, b2=0.999):
        self.lr, self.wd, self.b1, self.b2, self.t = lr, wd, b1, b2, 0
        self.p, self.m, self.v = [], [], []
        self._collect(model)

    def _collect(self, obj):
        if isinstance(obj, Linear):
            for param in [obj.W, obj.b]:
                self.p.append(param)
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))
        elif isinstance(obj, RMSNorm):
            self.p.append(obj.g)
            self.m.append(np.zeros_like(obj.g))
            self.v.append(np.zeros_like(obj.g))
        elif isinstance(obj, (list, tuple)):
            for i in obj: self._collect(i)
        elif hasattr(obj, "__dict__"):
            for k, v in obj.__dict__.items():
                if k not in ["rope", "x", "xn", "r", "p", "qr", "kr", "v_raw", "gw", "idx", "caches", "sh", "latent"]:
                    self._collect(v)

    def step(self, model):
        self.t += 1
        gs = []
        self._get_grads(model, gs)
        lr_t = self.lr * (np.sqrt(1 - self.b2**self.t) / (1 - self.b1**self.t))
        for i in range(len(self.p)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * gs[i]
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (gs[i]**2)
            self.p[i] -= lr_t * (self.m[i] / (np.sqrt(self.v[i]) + 1e-8) + self.wd * self.p[i])

    def _get_grads(self, obj, gs):
        if isinstance(obj, Linear): gs.extend([obj.dW, obj.db])
        elif isinstance(obj, RMSNorm): gs.append(obj.dg)
        elif isinstance(obj, (list, tuple)):
            for i in obj: self._get_grads(i, gs)
        elif hasattr(obj, "__dict__"):
            for k, v in obj.__dict__.items():
                if k not in ["rope", "x", "xn", "r", "p", "qr", "kr", "v_raw", "gw", "idx", "caches", "sh", "latent"]:
                    self._get_grads(v, gs)

def train():
    N, D, C, BS, E = 512, 784, 10, 64, 100
    X = np.random.randn(N, D).astype("f4")
    Y = np.random.randint(0, C, N)
    
    model = OMEGA_ASI(D, 256, C, depth=3)
    opt = AdamW(model, lr=1e-3, wd=0.05)
    
    for epoch in range(E):
        indices = np.random.permutation(N)
        total_loss, correct = 0, 0
        
        for i in range(0, N, BS):
            batch_idx = indices[i:i+BS]
            xb, yb = X[batch_idx], Y[batch_idx]
            
            logits = model.forward(xb)
            
            # Softmax Cross-Entropy
            exps = np.exp(logits - np.max(logits, -1, keepdims=True))
            probs = exps / np.sum(exps, -1, keepdims=True)
            
            batch_loss = -np.mean(np.log(probs[np.arange(len(yb)), yb] + 1e-12))
            total_loss += batch_loss * len(yb)
            correct += np.sum(np.argmax(probs, -1) == yb)
            
            d_logits = probs.copy()
            d_logits[np.arange(len(yb)), yb] -= 1
            model.backward(d_logits / len(yb))
            opt.step(model)
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d} | Loss: {total_loss/N:.4f} | Acc: {correct/N:.4f}")

if __name__ == "__main__":
    train()
