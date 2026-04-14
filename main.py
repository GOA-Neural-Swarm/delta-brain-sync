import numpy as np
import time

def swiglu(x):
    h = x.shape[-1] // 2
    a, b = x[..., :h], x[..., h:]
    return a * (1.0 / (1.0 + np.exp(-np.clip(a, -10, 10)))) * b

def d_swiglu(x, dout):
    h = x.shape[-1] // 2
    a, b = x[..., :h], x[..., h:]
    s = 1.0 / (1.0 + np.exp(-np.clip(a, -10, 10)))
    sw = a * s
    da = dout * b * (s + sw * (1.0 - s))
    db = dout * sw
    return np.concatenate([da, db], axis=-1)

class Linear:
    def __init__(self, in_d, out_d, init_scale=1.0):
        self.W = np.random.randn(in_d, out_d).astype(np.float32) * (np.sqrt(2.0 / in_d) * init_scale)
        self.b = np.zeros(out_d, dtype=np.float32)
        self.x, self.dW, self.db = None, None, None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.reshape(-1, self.x.shape[-1]).T @ dout.reshape(-1, dout.shape[-1])
        self.db = np.sum(dout, axis=tuple(range(len(dout.shape)-1)))
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
        inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
        t = np.arange(max_seq)
        freqs = np.outer(t, inv_freq)
        self.cos = np.cos(freqs)[None, :, None, :]
        self.sin = np.sin(freqs)[None, :, None, :]

    def apply(self, x, rev=False):
        b, s, h, d = x.shape
        d2 = d // 2
        x1, x2 = x[..., :d2], x[..., d2:]
        c, sn = self.cos[:, :s, :, :], self.sin[:, :s, :, :]
        if rev: return np.concatenate([x1 * c + x2 * sn, x2 * c - x1 * sn], axis=-1)
        return np.concatenate([x1 * c - x2 * sn, x2 * c + x1 * sn], axis=-1)

class Attention:
    def __init__(self, dim, heads=8, kv_heads=4):
        self.dim, self.h, self.kv = dim, heads, kv_heads
        self.hd = dim // heads
        self.g = heads // kv_heads
        self.wq = Linear(dim, dim)
        self.wk = Linear(dim, kv_heads * self.hd)
        self.wv = Linear(dim, kv_heads * self.hd)
        self.wo = Linear(dim, dim)
        self.rope = RoPE(self.hd)
        self.scale = 1.0 / np.sqrt(self.hd)

    def forward(self, x):
        b, s, _ = x.shape
        q = self.wq.forward(x).reshape(b, s, self.h, self.hd)
        k = self.wk.forward(x).reshape(b, s, self.kv, self.hd)
        v = self.wv.forward(x).reshape(b, s, self.kv, self.hd)
        self.q, self.k = self.rope.apply(q), self.rope.apply(k)
        self.v = v
        k_rep = np.repeat(self.k, self.g, axis=2)
        v_rep = np.repeat(self.v, self.g, axis=2)
        attn = np.einsum("bshd,bthd->bsht", self.q, k_rep) * self.scale
        attn_max = np.max(attn, axis=-1, keepdims=True)
        exp_a = np.exp(attn - attn_max)
        self.probs = exp_a / (np.sum(exp_a, axis=-1, keepdims=True) + 1e-10)
        out = np.einsum("bsht,bthd->bshd", self.probs, v_rep)
        return self.wo.forward(out.reshape(b, s, self.dim))

    def backward(self, dout):
        b, s, _ = dout.shape
        dout_o = self.wo.backward(dout).reshape(b, s, self.h, self.hd)
        k_rep = np.repeat(self.k, self.g, axis=2)
        v_rep = np.repeat(self.v, self.g, axis=2)
        dv_rep = np.einsum("bsht,bshd->bthd", self.probs, dout_o)
        dprobs = np.einsum("bshd,bthd->bsht", dout_o, v_rep)
        dattn = self.probs * (dprobs - np.sum(self.probs * dprobs, axis=-1, keepdims=True)) * self.scale
        dq = np.einsum("bsht,bthd->bshd", dattn, k_rep)
        dk_rep = np.einsum("bsht,bshd->bthd", dattn, self.q)
        dq = self.rope.apply(dq, rev=True)
        dk = self.rope.apply(np.sum(dk_rep.reshape(b, s, self.kv, self.g, self.hd), axis=3), rev=True)
        dv = np.sum(dv_rep.reshape(b, s, self.kv, self.g, self.hd), axis=3)
        return self.wq.backward(dq.reshape(b, s, -1)) + self.wk.backward(dk.reshape(b, s, -1)) + self.wv.backward(dv.reshape(b, s, -1))

class RedundantLogicGate:
    def __init__(self, dim):
        self.gemini_path = [Linear(dim, dim * 2), Linear(dim * 2, dim)]
        self.groq_path = [Linear(dim, dim * 2), Linear(dim * 2, dim)]
        self.gate = Linear(dim, 2)

    def forward(self, x):
        self.x = x
        self.h_gem = swiglu(self.gemini_path[0].forward(x))
        self.out_gem = self.gemini_path[1].forward(self.h_gem)
        self.h_groq = swiglu(self.groq_path[0].forward(x))
        self.out_groq = self.groq_path[1].forward(self.h_groq)
        g = self.gate.forward(x)
        self.p = np.exp(g - np.max(g, -1, keepdims=True))
        self.p /= np.sum(self.p, -1, keepdims=True) + 1e-10
        return self.p[..., 0:1] * self.out_gem + self.p[..., 1:2] * self.out_groq

    def backward(self, dout):
        dp = np.stack([np.sum(dout * self.out_gem, -1), np.sum(dout * self.out_groq, -1)], axis=-1)
        dg = self.p * (dp - np.sum(self.p * dp, -1, keepdims=True))
        dx_gate = self.gate.backward(dg)
        dgem = self.gemini_path[1].backward(dout * self.p[..., 0:1])
        dgem = self.gemini_path[0].backward(d_swiglu(self.gemini_path[0].x, dgem))
        dgroq = self.groq_path[1].backward(dout * self.p[..., 1:2])
        dgroq = self.groq_path[0].backward(d_swiglu(self.groq_path[0].x, dgroq))
        return dx_gate + dgem + dgroq

class MoE:
    def __init__(self, dim, num_exp=4):
        self.dim, self.num_exp = dim, num_exp
        self.gate = Linear(dim, num_exp)
        self.experts = [[Linear(dim, dim*2), Linear(dim*2, dim)] for _ in range(num_exp)]

    def forward(self, x):
        orig_shape = x.shape
        x = x.reshape(-1, self.dim)
        logits = self.gate.forward(x)
        self.p = np.exp(logits - np.max(logits, -1, keepdims=True))
        self.p /= np.sum(self.p, -1, keepdims=True) + 1e-10
        self.sel = np.argmax(self.p, axis=-1)
        out = np.zeros_like(x)
        self.cache = []
        for i in range(self.num_exp):
            mask = (self.sel == i)
            if not np.any(mask):
                self.cache.append(None)
                continue
            h = swiglu(self.experts[i][0].forward(x[mask]))
            y = self.experts[i][1].forward(h)
            out[mask] = y * self.p[mask, i:i+1]
            self.cache.append((mask, h, y))
        return out.reshape(orig_shape)

    def backward(self, dout):
        orig_shape = dout.shape
        dout = dout.reshape(-1, self.dim)
        dx = np.zeros_like(dout)
        dp = np.zeros_like(self.p)
        for i in range(self.num_exp):
            if self.cache[i] is None: continue
            mask, h, y = self.cache[i]
            dp[mask, i] = np.sum(dout[mask] * y, axis=-1)
            dy = dout[mask] * self.p[mask, i:i+1]
            dh = self.experts[i][1].backward(dy)
            dx[mask] += self.experts[i][0].backward(d_swiglu(self.experts[i][0].x, dh))
        dg = self.p * (dp - np.sum(self.p * dp, -1, keepdims=True))
        return (dx + self.gate.backward(dg)).reshape(orig_shape)

class OMEGA_Block:
    def __init__(self, dim):
        self.ln1 = RMSNorm(dim)
        self.attn = Attention(dim)
        self.ln2 = RMSNorm(dim)
        self.logic = RedundantLogicGate(dim)
        self.ln3 = RMSNorm(dim)
        self.moe = MoE(dim)

    def forward(self, x):
        x = x + self.attn.forward(self.ln1.forward(x))
        x = x + self.logic.forward(self.ln2.forward(x))
        x = x + self.moe.forward(self.ln3.forward(x))
        return x

    def backward(self, dout):
        dout = dout + self.ln3.backward(self.moe.backward(dout))
        dout = dout + self.ln2.backward(self.logic.backward(dout))
        dout = dout + self.ln1.backward(self.attn.backward(dout))
        return dout

class OMEGA_ASI_X5:
    def __init__(self, in_dim=784, h_dim=128, out_dim=10, depth=2):
        self.stem = Linear(in_dim, h_dim)
        self.blocks = [OMEGA_Block(h_dim) for _ in range(depth)]
        self.norm = RMSNorm(h_dim)
        self.head = Linear(h_dim, out_dim)

    def forward(self, x):
        x = self.stem.forward(x)[:, None, :]
        for b in self.blocks: x = b.forward(x)
        self.feat = self.norm.forward(x[:, 0, :])
        return self.head.forward(self.feat)

    def backward(self, dout):
        dout = self.norm.backward(self.head.backward(dout))[:, None, :]
        for b in reversed(self.blocks): dout = b.backward(dout)
        self.stem.backward(dout[:, 0, :])

    def get_params(self):
        params = []
        def find(obj):
            if isinstance(obj, (Linear, RMSNorm)): params.append(obj)
            elif isinstance(obj, list): [find(i) for i in obj]
            elif hasattr(obj, "__dict__"): [find(v) for v in obj.__dict__.values()]
        find(self)
        return params

class LionOptimizer:
    def __init__(self, params, lr=1e-4, b1=0.9, b2=0.99, wd=0.01):
        self.params, self.lr, self.b1, self.b2, self.wd = params, lr, b1, b2, wd
        self.m = [np.zeros_like(p.W if hasattr(p, 'W') else p.g) for p in params]
        self.mb = [np.zeros_like(p.b) if hasattr(p, 'b') else None for p in params]

    def step(self):
        for i, p in enumerate(self.params):
            if hasattr(p, 'W'):
                for attr, mom in [('W', self.m), ('b', self.mb)]:
                    if mom[i] is None: continue
                    grad, w = getattr(p, 'd'+attr), getattr(p, attr)
                    update = np.sign(self.b1 * mom[i] + (1.0 - self.b1) * grad)
                    w -= self.lr * (update + self.wd * w if attr == 'W' else update)
                    mom[i] = self.b2 * mom[i] + (1.0 - self.b2) * grad
                    setattr(p, attr, w)
            else:
                update = np.sign(self.b1 * self.m[i] + (1.0 - self.b1) * p.dg)
                p.g -= self.lr * (update + self.wd * p.g)
                self.m[i] = self.b2 * self.m[i] + (1.0 - self.b2) * p.dg

def train_omega():
    N, D, C = 2048, 784, 10
    X = np.random.randn(N, D).astype(np.float32)
    y = np.random.randint(0, C, N)
    model = OMEGA_ASI_X5(in_dim=D, h_dim=64, out_dim=C, depth=1)
    optimizer = LionOptimizer(model.get_params(), lr=3e-4)
    batch_size = 64
    
    for epoch in range(20):
        indices = np.random.permutation(N)
        total_loss, correct = 0, 0
        t0 = time.time()
        for i in range(0, N, batch_size):
            idx = indices[i:i+batch_size]
            xb, yb = X[idx], y[idx]
            
            logits = model.forward(xb)
            probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs /= np.sum(probs, axis=1, keepdims=True) + 1e-10
            
            loss = -np.mean(np.log(probs[range(len(yb)), yb] + 1e-10))
            total_loss += loss * len(yb)
            correct += np.sum(np.argmax(probs, axis=1) == yb)
            
            dout = probs.copy()
            dout[range(len(yb)), yb] -= 1
            model.backward(dout / len(yb))
            
            gnorm = np.sqrt(sum(np.sum(getattr(p, 'dW', 0)**2) + np.sum(getattr(p, 'db', 0)**2) + np.sum(getattr(p, 'dg', 0)**2) for p in model.get_params()))
            if gnorm > 5.0:
                for p in model.get_params():
                    if hasattr(p, 'dW'): p.dW *= 5.0/gnorm; p.db *= 5.0/gnorm
                    if hasattr(p, 'dg'): p.dg *= 5.0/gnorm
            
            optimizer.step()
            
        print(f"Epoch {epoch} | Loss: {total_loss/N:.4f} | Acc: {correct/N:.4f} | Time: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    train_omega()
