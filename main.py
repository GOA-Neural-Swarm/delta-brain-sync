import numpy as np
import time

def swiglu(x):
    h = x.shape[-1] // 2
    a, b = x[..., :h], x[..., h:]
    sig = 1.0 / (1.0 + np.exp(-np.clip(a, -12, 12)))
    return (a * sig) * b

def d_swiglu(x, dout):
    h = x.shape[-1] // 2
    a, b = x[..., :h], x[..., h:]
    sig = 1.0 / (1.0 + np.exp(-np.clip(a, -12, 12)))
    swi = a * sig
    da = dout * b * (sig + swi * (1.0 - sig))
    db = dout * swi
    return np.concatenate([da, db], axis=-1)

def softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)

class Linear:
    def __init__(self, in_d, out_d, init_type='kaiming'):
        scale = np.sqrt(2.0 / in_d) if init_type == 'kaiming' else 0.02
        self.W = np.random.normal(0, scale, (in_d, out_d)).astype(np.float32)
        self.b = np.zeros(out_d, dtype=np.float32)
        self.x, self.dW, self.db = None, None, None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        x_flat = self.x.reshape(-1, self.x.shape[-1])
        dout_flat = dout.reshape(-1, dout.shape[-1])
        self.dW = np.dot(x_flat.T, dout_flat)
        self.db = np.sum(dout_flat, axis=0)
        return np.dot(dout, self.W.T)

class RMSNorm:
    def __init__(self, dim, eps=1e-6):
        self.g = np.ones(dim, dtype=np.float32)
        self.eps = eps
        self.x, self.inv_rms = None, None

    def forward(self, x):
        self.x = x
        self.inv_rms = 1.0 / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return self.g * (x * self.inv_rms)

    def backward(self, dout):
        nx = self.x * self.inv_rms
        self.dg = np.sum(dout * nx, axis=tuple(range(len(dout.shape)-1)))
        d_nx = dout * self.g
        return self.inv_rms * (d_nx - nx * np.mean(d_nx * nx, axis=-1, keepdims=True))

class RoPE:
    def __init__(self, dim, max_seq=2048):
        inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2).astype(np.float32) / dim))
        t = np.arange(max_seq)
        freqs = np.outer(t, inv_freq)
        emb = np.concatenate([freqs, freqs], axis=-1)
        self.cos, self.sin = np.cos(emb)[None, :, None, :], np.sin(emb)[None, :, None, :]

    def apply(self, x):
        s = x.shape[1]
        c, sn = self.cos[:, :s, :, :], self.sin[:, :s, :, :]
        h = x.shape[-1] // 2
        x_rot = np.concatenate([-x[..., h:], x[..., :h]], axis=-1)
        return x * c + x_rot * sn

    def backward(self, dout):
        s = dout.shape[1]
        c, sn = self.cos[:, :s, :, :], self.sin[:, :s, :, :]
        h = dout.shape[-1] // 2
        dout_rot = np.concatenate([dout[..., h:], -dout[..., :h]], axis=-1)
        return dout * c + dout_rot * sn

class SovereignAttention:
    def __init__(self, dim, heads=8, kv_heads=2):
        self.dim, self.heads, self.kv_heads = dim, heads, kv_heads
        self.h_dim = dim // heads
        self.group = heads // kv_heads
        self.q_proj = Linear(dim, dim)
        self.k_proj = Linear(dim, kv_heads * self.h_dim)
        self.v_proj = Linear(dim, kv_heads * self.h_dim)
        self.o_proj = Linear(dim, dim)
        self.rope = RoPE(self.h_dim)
        self.scale = 1.0 / np.sqrt(self.h_dim)

    def forward(self, x):
        b, s, d = x.shape
        self.q = self.q_proj.forward(x).reshape(b, s, self.heads, self.h_dim)
        self.k = self.k_proj.forward(x).reshape(b, s, self.kv_heads, self.h_dim)
        self.v = self.v_proj.forward(x).reshape(b, s, self.kv_heads, self.h_dim)
        self.qr, self.kr = self.rope.apply(self.q), self.rope.apply(self.k)
        self.k_rep = np.repeat(self.kr, self.group, axis=2)
        self.v_rep = np.repeat(self.v, self.group, axis=2)
        attn = np.einsum("bshd,bthd->bsht", self.qr, self.k_rep) * self.scale
        self.probs = softmax(attn)
        out = np.einsum("bsht,bthd->bshd", self.probs, self.v_rep)
        return self.o_proj.forward(out.reshape(b, s, d))

    def backward(self, dout):
        b, s, d = dout.shape
        dout_o = self.o_proj.backward(dout).reshape(b, s, self.heads, self.h_dim)
        dv_rep = np.einsum("bsht,bshd->bthd", self.probs, dout_o)
        d_probs = np.einsum("bshd,bthd->bsht", dout_o, self.v_rep)
        d_attn = self.probs * (d_probs - np.sum(self.probs * d_probs, axis=-1, keepdims=True)) * self.scale
        d_qr = np.einsum("bsht,bthd->bshd", d_attn, self.k_rep)
        d_k_rep = np.einsum("bsht,bshd->bthd", d_attn, self.qr)
        dq = self.rope.backward(d_qr)
        dk = self.rope.backward(np.sum(d_k_rep.reshape(b, s, self.kv_heads, self.group, self.h_dim), axis=3))
        dv = np.sum(dv_rep.reshape(b, s, self.kv_heads, self.group, self.h_dim), axis=3)
        return self.q_proj.backward(dq.reshape(b, s, -1)) + self.k_proj.backward(dk.reshape(b, s, -1)) + self.v_proj.backward(dv.reshape(b, s, -1))

class GeminiGroqConsensus:
    def __init__(self, dim):
        self.gemini_path = Linear(dim, dim * 2)
        self.gemini_out = Linear(dim * 2, dim)
        self.groq_path = Linear(dim, dim)
        self.gate = Linear(dim, 2)

    def forward(self, x):
        self.x = x
        self.g_mid = swiglu(self.gemini_path.forward(x))
        self.g1 = self.gemini_out.forward(self.g_mid)
        self.g2 = self.groq_path.forward(x)
        self.logits = self.gate.forward(x)
        self.probs = softmax(self.logits)
        return self.probs[..., 0:1] * self.g1 + self.probs[..., 1:2] * self.g2

    def backward(self, dout):
        dg1 = dout * self.probs[..., 0:1]
        dg2 = dout * self.probs[..., 1:2]
        d_p1 = np.sum(dout * self.g1, axis=-1, keepdims=True)
        d_p2 = np.sum(dout * self.g2, axis=-1, keepdims=True)
        d_probs = np.concatenate([d_p1, d_p2], axis=-1)
        d_logits = self.probs * (d_probs - np.sum(self.probs * d_probs, axis=-1, keepdims=True))
        
        dx_g1 = self.gemini_path.backward(d_swiglu(self.gemini_path.forward(self.x), self.gemini_out.backward(dg1)))
        dx_g2 = self.groq_path.backward(dg2)
        dx_gate = self.gate.backward(d_logits)
        return dx_g1 + dx_g2 + dx_gate

class SovereignMoE:
    def __init__(self, dim, n_exp=4, mult=2):
        self.dim, self.n_exp = dim, n_exp
        self.gate = Linear(dim, n_exp)
        self.experts = [[Linear(dim, dim * mult), Linear(dim * mult, dim)] for _ in range(n_exp)]

    def forward(self, x):
        self.shape = x.shape
        xf = x.reshape(-1, self.dim)
        self.probs = softmax(self.gate.forward(xf))
        self.indices = np.argmax(self.probs, axis=-1)
        out = np.zeros_like(xf)
        self.ex_cache = []
        for i in range(self.n_exp):
            mask = (self.indices == i)
            if not np.any(mask):
                self.ex_cache.append(None)
                continue
            h = self.experts[i][0].forward(xf[mask])
            act = swiglu(h)
            eo = self.experts[i][1].forward(act)
            out[mask] = eo * self.probs[mask, i][:, None]
            self.ex_cache.append((mask, h, act, eo))
        return out.reshape(self.shape)

    def backward(self, dout):
        df = dout.reshape(-1, self.dim)
        d_logits = np.zeros_like(self.probs)
        dx = np.zeros((df.shape[0], self.dim), dtype=np.float32)
        for i in range(self.n_exp):
            if self.ex_cache[i] is None: continue
            mask, h, act, eo = self.ex_cache[i]
            d_eo = df[mask] * self.probs[mask, i][:, None]
            d_logits[mask, i] = np.sum(df[mask] * eo, axis=-1)
            d_act = self.experts[i][1].backward(d_eo)
            d_h = d_swiglu(h, d_act)
            dx[mask] += self.experts[i][0].backward(d_h)
        dg = self.probs * (d_logits - np.sum(self.probs * d_logits, axis=-1, keepdims=True))
        return (dx + self.gate.backward(dg)).reshape(self.shape)

class SovereignBlock:
    def __init__(self, dim):
        self.n1, self.attn = RMSNorm(dim), SovereignAttention(dim)
        self.n2, self.consensus = RMSNorm(dim), GeminiGroqConsensus(dim)
        self.n3, self.moe = RMSNorm(dim), SovereignMoE(dim)

    def forward(self, x):
        x = x + self.attn.forward(self.n1.forward(x))
        x = x + self.consensus.forward(self.n2.forward(x))
        x = x + self.moe.forward(self.n3.forward(x))
        return x

    def backward(self, dout):
        dm = self.moe.backward(dout)
        dx = dout + self.n3.backward(dm)
        dc = self.consensus.backward(dx)
        dx = dx + self.n2.backward(dc)
        da = self.attn.backward(dx)
        return dx + self.n1.backward(da)

class OMEGA_ASI_X2:
    def __init__(self, in_d=784, h_d=128, out_d=10, depth=2):
        self.stem = Linear(in_d, h_d)
        self.blocks = [SovereignBlock(h_d) for _ in range(depth)]
        self.norm = RMSNorm(h_d)
        self.head = Linear(h_d, out_d)

    def forward(self, x):
        x = self.stem.forward(x)[:, None, :]
        for b in self.blocks: x = b.forward(x)
        self.f = self.norm.forward(x[:, 0, :])
        return self.head.forward(self.f)

    def backward(self, dout):
        dout = self.norm.backward(self.head.backward(dout))[:, None, : ]
        for b in reversed(self.blocks): dout = b.backward(dout)
        self.stem.backward(dout[:, 0, :])

    def get_params(self):
        p = []
        def coll(o):
            if isinstance(o, (Linear, RMSNorm)): p.append(o)
            elif hasattr(o, "__dict__"):
                for v in o.__dict__.values():
                    if isinstance(v, list): [coll(i) for i in v]
                    else: coll(v)
        coll(self)
        return p

class Lion:
    def __init__(self, params, lr=1e-4, b1=0.9, b2=0.99, wd=0.01):
        self.params, self.lr, self.b1, self.b2, self.wd = params, lr, b1, b2, wd
        self.m = [np.zeros_like(getattr(p, "W", getattr(p, "g", None))) for p in params]
        self.mb = [np.zeros_like(p.b) if hasattr(p, "b") else None for p in params]

    def step(self):
        for i, p in enumerate(self.params):
            if hasattr(p, "W"):
                for attr, m_list in [("W", self.m), ("b", self.mb)]:
                    if m_list[i] is None: continue
                    g, m, w = getattr(p, "d" + attr), m_list[i], getattr(p, attr)
                    u = np.sign(self.b1 * m + (1.0 - self.b1) * g)
                    w -= self.lr * (u + self.wd * w if attr == "W" else u)
                    m_list[i] = self.b2 * m + (1.0 - self.b2) * g
                    setattr(p, attr, w)
            else:
                m, g = self.m[i], p.dg
                u = np.sign(self.b1 * m + (1.0 - self.b1) * g)
                p.g -= self.lr * (u + self.wd * p.g)
                self.m[i] = self.b2 * m + (1.0 - self.b2) * g

def get_data(n=2048):
    X = np.random.randn(n, 784).astype(np.float32)
    y = np.random.randint(0, 10, n)
    for i in range(n): X[i, y[i]*78:(y[i]+1)*78] += 4.0
    return (X - np.mean(X)) / (np.std(X) + 1e-6), y

def train():
    X, y = get_data(4096)
    model = OMEGA_ASI_X2(h_d=64, depth=1)
    params = model.get_params()
    opt = Lion(params, lr=2e-4)
    bs, epochs = 64, 50
    for ep in range(epochs):
        idx = np.random.permutation(len(X))
        ls, acc, t0 = 0, 0, time.time()
        opt.lr = 2e-4 * 0.5 * (1 + np.cos(np.pi * ep / epochs))
        for i in range(0, len(X), bs):
            xb, yb = X[idx[i:i+bs]], y[idx[i:i+bs]]
            if len(xb) < bs: continue
            logits = model.forward(xb)
            probs = softmax(logits)
            ls += -np.mean(np.log(probs[range(bs), yb] + 1e-10)) * bs
            acc += np.sum(np.argmax(probs, axis=1) == yb)
            dout = probs.copy(); dout[range(bs), yb] -= 1
            model.backward(dout / bs)
            gn = np.sqrt(sum(np.sum(p.dW**2) + np.sum(p.db**2) if hasattr(p, "dW") else np.sum(p.dg**2) for p in params))
            if gn > 1.0:
                for p in params:
                    if hasattr(p, "dW"): p.dW /= gn; p.db /= gn
                    if hasattr(p, "dg"): p.dg /= gn
            opt.step()
        print(f"EP:{ep:02d} | LOSS:{ls/len(X):.4f} | ACC:{acc/len(X):.4f} | {len(X)/(time.time()-t0):.1f} s/s")

if __name__ == "__main__":
    train()
