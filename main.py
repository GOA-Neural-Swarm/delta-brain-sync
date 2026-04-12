import numpy as np
import time

def silu(x):
    return x / (1.0 + np.exp(-x))

def d_silu(x, dout):
    sig = 1.0 / (1.0 + np.exp(-x))
    return dout * (sig * (1.0 + x * (1.0 - sig)))

def fast_softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exps = np.exp(x - x_max)
    return exps / (np.sum(exps, axis=axis, keepdims=True) + 1e-10)

class Linear:
    def __init__(self, in_d, out_d, std=0.02):
        self.W = np.random.normal(0, std, (in_d, out_d)).astype(np.float32)
        self.b = np.zeros(out_d, dtype=np.float32)
        self.dW, self.db = None, None
        self.x = None

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
        self.scale = np.ones(dim, dtype=np.float32)
        self.eps = eps
        self.x, self.rstd = None, None

    def forward(self, x):
        self.x = x
        self.rstd = 1.0 / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return self.scale * (x * self.rstd)

    def backward(self, dout):
        norm_x = self.x * self.rstd
        self.d_scale = np.sum(dout * norm_x, axis=tuple(range(len(dout.shape) - 1)))
        v = dout * self.scale
        return self.rstd * (v - norm_x * np.mean(v * norm_x, axis=-1, keepdims=True))

class RotaryEmbedding:
    def __init__(self, dim, max_len=2048):
        inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2).astype(np.float32) / dim))
        t = np.arange(max_len, dtype=np.float32)
        freqs = np.outer(t, inv_freq)
        self.cos = np.cos(np.concatenate((freqs, freqs), axis=-1))
        self.sin = np.sin(np.concatenate((freqs, freqs), axis=-1))

    def apply(self, x):
        s = x.shape[1]
        c, s_ = self.cos[:s][None, :, None, :], self.sin[:s][None, :, None, :]
        half = x.shape[-1] // 2
        x_rot = np.concatenate((-x[..., half:], x[..., :half]), axis=-1)
        return x * c + x_rot * s_

    def backward(self, dout):
        s = dout.shape[1]
        c, s_ = self.cos[:s][None, :, None, :], self.sin[:s][None, :, None, :]
        half = dout.shape[-1] // 2
        dx = dout * c
        dx[..., :half] += dout[..., half:] * s_[..., :half]
        dx[..., half:] -= dout[..., :half] * s_[..., half:]
        return dx

class GeminiGQA:
    def __init__(self, dim, heads=8, kv_heads=2):
        self.dim, self.heads, self.kv_heads = dim, heads, kv_heads
        self.head_dim = dim // heads
        self.group = heads // kv_heads
        self.q_proj = Linear(dim, dim)
        self.k_proj = Linear(dim, kv_heads * self.head_dim)
        self.v_proj = Linear(dim, kv_heads * self.head_dim)
        self.o_proj = Linear(dim, dim)
        self.rope = RotaryEmbedding(self.head_dim)
        self.scale = 1.0 / np.sqrt(self.head_dim)

    def forward(self, x):
        b, s, d = x.shape
        q = self.q_proj.forward(x).reshape(b, s, self.heads, self.head_dim)
        k = self.k_proj.forward(x).reshape(b, s, self.kv_heads, self.head_dim)
        v = self.v_proj.forward(x).reshape(b, s, self.kv_heads, self.head_dim)
        self.q_rope, self.k_rope = self.rope.apply(q), self.rope.apply(k)
        self.v_val = v
        k_rep = np.repeat(self.k_rope, self.group, axis=2)
        v_rep = np.repeat(v, self.group, axis=2)
        attn = np.einsum("bshd,bthd->bsht", self.q_rope, k_rep) * self.scale
        self.probs = fast_softmax(attn)
        out = np.einsum("bsht,bthd->bshd", self.probs, v_rep)
        return self.o_proj.forward(out.reshape(b, s, d))

    def backward(self, dout):
        b, s, d = dout.shape
        dout_o = self.o_proj.backward(dout).reshape(b, s, self.heads, self.head_dim)
        k_rep = np.repeat(self.k_rope, self.group, axis=2)
        v_rep = np.repeat(self.v_val, self.group, axis=2)
        d_probs = np.einsum("bshd,bthd->bsht", dout_o, v_rep)
        d_v_rep = np.einsum("bsht,bshd->bthd", self.probs, dout_o)
        d_attn = self.probs * (d_probs - np.sum(self.probs * d_probs, axis=-1, keepdims=True)) * self.scale
        dq_rope = np.einsum("bsht,bthd->bshd", d_attn, k_rep)
        dk_rep = np.einsum("bsht,bshd->bthd", d_attn, self.q_rope)
        dq = self.rope.backward(dq_rope).reshape(b, s, d)
        dk_rope = np.sum(dk_rep.reshape(b, s, self.kv_heads, self.group, self.head_dim), axis=3)
        dk = self.rope.backward(dk_rope).reshape(b, s, -1)
        dv = np.sum(d_v_rep.reshape(b, s, self.kv_heads, self.group, self.head_dim), axis=3).reshape(b, s, -1)
        return self.q_proj.backward(dq) + self.k_proj.backward(dk) + self.v_proj.backward(dv)

class GroqMoE:
    def __init__(self, dim, num_experts=4, top_k=1):
        self.dim, self.num_experts, self.top_k = dim, num_experts, top_k
        self.gate = Linear(dim, num_experts)
        self.experts = [[Linear(dim, dim * 2), Linear(dim * 2, dim)] for _ in range(num_experts)]

    def forward(self, x):
        self.orig_shape = x.shape
        x_flat = x.reshape(-1, self.dim)
        logits = self.gate.forward(x_flat)
        probs = fast_softmax(logits)
        idx = np.argmax(probs, axis=-1)
        self.idx, self.probs = idx, probs
        out = np.zeros_like(x_flat)
        self.ex_ctx = []
        for i in range(self.num_experts):
            mask = (idx == i)
            if not np.any(mask):
                self.ex_ctx.append(None)
                continue
            e_in = x_flat[mask]
            h = self.experts[i][0].forward(e_in)
            act = silu(h)
            e_out = self.experts[i][1].forward(act)
            out[mask] = e_out * probs[mask, i:i+1]
            self.ex_ctx.append((mask, h, act, e_out))
        return out.reshape(self.orig_shape)

    def backward(self, dout):
        dout_flat = dout.reshape(-1, self.dim)
        dx = np.zeros((dout_flat.shape[0], self.dim), dtype=np.float32)
        d_logits = np.zeros_like(self.probs)
        for i in range(self.num_experts):
            if self.ex_ctx[i] is None: continue
            mask, h, act, e_out = self.ex_ctx[i]
            p = self.probs[mask, i:i+1]
            de_out = dout_flat[mask] * p
            d_logits[mask, i] = np.sum(dout_flat[mask] * e_out, axis=-1)
            d_act = self.experts[i][1].backward(de_out)
            dh = d_silu(h, d_act)
            dx[mask] += self.experts[i][0].backward(dh)
        dg = self.probs * (d_logits - np.sum(self.probs * d_logits, axis=-1, keepdims=True))
        dx += self.gate.backward(dg)
        return dx.reshape(self.orig_shape)

class SovereignBlock:
    def __init__(self, dim):
        self.norm1 = RMSNorm(dim)
        self.attn = GeminiGQA(dim)
        self.norm2 = RMSNorm(dim)
        self.moe = GroqMoE(dim)

    def forward(self, x):
        x = x + self.attn.forward(self.norm1.forward(x))
        x = x + self.moe.forward(self.norm2.forward(x))
        return x

    def backward(self, dout):
        dm = self.moe.backward(dout)
        dout = dout + self.norm2.backward(dm)
        da = self.attn.backward(dout)
        return dout + self.norm1.backward(da)

class OMEGA_ASI_V8:
    def __init__(self, in_dim=784, h_dim=128, out_dim=10, depth=2):
        self.stem = Linear(in_dim, h_dim)
        self.blocks = [SovereignBlock(h_dim) for _ in range(depth)]
        self.norm = RMSNorm(h_dim)
        self.head = Linear(h_dim, out_dim)

    def forward(self, x):
        x = self.stem.forward(x)[:, None, :]
        for b in self.blocks: x = b.forward(x)
        self.feat = self.norm.forward(x[:, 0, :])
        return self.head.forward(self.feat)

    def backward(self, dout):
        dout = self.head.backward(dout)
        dout = self.norm.backward(dout)[:, None, :]
        for b in reversed(self.blocks): dout = b.backward(dout)
        self.stem.backward(dout[:, 0, :])

    def get_params(self):
        p = []
        def collect(obj):
            if isinstance(obj, (Linear, RMSNorm)): p.append(obj)
            elif hasattr(obj, "__dict__"):
                for v in obj.__dict__.values():
                    if isinstance(v, list): [collect(i) for i in v]
                    else: collect(v)
        collect(self)
        return p

class Lion:
    def __init__(self, params, lr=1e-4, b1=0.9, b2=0.99, wd=0.01):
        self.params, self.lr, self.b1, self.b2, self.wd = params, lr, b1, b2, wd
        self.m = [np.zeros_like(p.W) if hasattr(p, 'W') else np.zeros_like(p.scale) for p in params]
        self.mb = [np.zeros_like(p.b) if hasattr(p, 'b') else None for p in params]

    def step(self):
        for i, p in enumerate(self.params):
            if hasattr(p, 'W'):
                for attr, m_list in [('W', self.m), ('b', self.mb)]:
                    g = getattr(p, 'd'+attr)
                    m = m_list[i]
                    u = np.sign(self.b1 * m + (1.0 - self.b1) * g)
                    w = getattr(p, attr)
                    w -= self.lr * (u + self.wd * w if attr == 'W' else u)
                    m_list[i] = self.b2 * m + (1.0 - self.b2) * g
                    setattr(p, attr, w)
            else:
                g = p.d_scale
                u = np.sign(self.b1 * self.m[i] + (1.0 - self.b1) * g)
                p.scale -= self.lr * (u + self.wd * p.scale)
                self.m[i] = self.b2 * self.m[i] + (1.0 - self.b2) * g

def get_data(n=2048):
    X = np.random.randn(n, 784).astype(np.float32)
    y = np.random.randint(0, 10, n)
    for i in range(n): X[i, y[i]*78:(y[i]+1)*78] += 5.0
    return (X - np.mean(X)) / (np.std(X) + 1e-6), y

def train():
    X, y = get_data(4096)
    model = OMEGA_ASI_V8(h_dim=128, depth=2)
    params = model.get_params()
    opt = Lion(params, lr=2e-4)
    bs = 64
    print("OMEGA-ASI V8 | RECURSIVE SELF-EVOLUTION | GQA-MOE HYBRID")
    for ep in range(30):
        idx = np.random.permutation(len(X))
        l_sum, a_sum, t0 = 0, 0, time.time()
        for i in range(0, len(X), bs):
            xb, yb = X[idx[i:i+bs]], y[idx[i:i+bs]]
            if len(xb) < bs: continue
            probs = fast_softmax(model.forward(xb))
            l_sum += -np.mean(np.log(probs[range(bs), yb] + 1e-10)) * bs
            a_sum += np.sum(np.argmax(probs, axis=1) == yb)
            dout = probs.copy(); dout[range(bs), yb] -= 1
            model.backward(dout / bs)
            gn = np.sqrt(sum(np.sum(p.dW**2) + np.sum(p.db**2) for p in params if hasattr(p, 'dW')))
            if gn > 1.0:
                for p in params:
                    if hasattr(p, 'dW'): p.dW /= gn; p.db /= gn
                    if hasattr(p, 'd_scale'): p.d_scale /= gn
            opt.step()
        print(f"EP:{ep:02d} | LOSS:{l_sum/len(X):.4f} | ACC:{a_sum/len(X):.4f} | {len(X)/(time.time()-t0):.1f} s/s")

if __name__ == "__main__":
    train()
