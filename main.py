import numpy as np
import time

def fast_softmax(x, axis=-1):
    max_x = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - max_x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)

def swiglu(x):
    half = x.shape[-1] // 2
    gate, val = x[..., :half], x[..., half:]
    return (gate * (1.0 / (1.0 + np.exp(-gate)))) * val

def d_swiglu(x, dout):
    half = x.shape[-1] // 2
    gate, val = x[..., :half], x[..., half:]
    sig = 1.0 / (1.0 + np.exp(-gate))
    swish = gate * sig
    d_gate = (sig * (1.0 + gate * (1.0 - sig))) * val * dout
    d_val = swish * dout
    return np.concatenate([d_gate, d_val], axis=-1)

class Linear:
    def __init__(self, in_f, out_f, init_scale=1.0):
        limit = np.sqrt(6.0 / (in_f + out_f)) * init_scale
        self.W = np.random.uniform(-limit, limit, (in_f, out_f)).astype(np.float32)
        self.b = np.zeros(out_f, dtype=np.float32)
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
        self.x, self.rstd, self.dg = None, None, None

    def forward(self, x):
        self.x = x
        self.rstd = 1.0 / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return self.g * (x * self.rstd)

    def backward(self, dout):
        nx = self.x * self.rstd
        self.dg = np.sum(dout * nx, axis=tuple(range(len(dout.shape) - 1)))
        v = dout * self.g
        return self.rstd * (v - nx * np.mean(v * nx, axis=-1, keepdims=True))

class RotaryEmbedding:
    def __init__(self, dim, max_seq_len=1024):
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2).astype(np.float32) / dim))
        t = np.arange(max_seq_len, dtype=np.float32)
        freqs = np.outer(t, inv_freq)
        emb = np.concatenate((freqs, freqs), axis=-1)
        self.cos_cache = np.cos(emb)
        self.sin_cache = np.sin(emb)

    def forward(self, q, k):
        s = q.shape[1]
        cos, sin = self.cos_cache[:s][None, :, None, :], self.sin_cache[:s][None, :, None, :]
        self.cos, self.sin = cos, sin
        def rotate(x):
            half = self.dim // 2
            x_rot = np.concatenate((-x[..., half:], x[..., :half]), axis=-1)
            return x * cos + x_rot * sin
        return rotate(q), rotate(k)

    def backward(self, dq, dk):
        half = self.dim // 2
        def d_rotate(dout):
            d_x_rot = dout * self.sin
            dx = dout * self.cos
            dx[..., :half] += d_x_rot[..., half:]
            dx[..., half:] -= d_x_rot[..., :half]
            return dx
        return d_rotate(dq), d_rotate(dk)

class GeminiGQA:
    def __init__(self, dim, heads=8, kv_heads=2):
        self.dim, self.heads, self.kv_heads = dim, heads, kv_heads
        self.hd = dim // heads
        self.scale = 1.0 / np.sqrt(self.hd)
        self.wq = Linear(dim, dim)
        self.wk = Linear(dim, self.kv_heads * self.hd)
        self.wv = Linear(dim, self.kv_heads * self.hd)
        self.wo = Linear(dim, dim)
        self.rope = RotaryEmbedding(self.hd)

    def forward(self, x):
        b, s, d = x.shape
        q = self.wq.forward(x).reshape(b, s, self.heads, self.hd)
        k = self.wk.forward(x).reshape(b, s, self.kv_heads, self.hd)
        v = self.wv.forward(x).reshape(b, s, self.kv_heads, self.hd)
        self.q_rot, self.k_rot = self.rope.forward(q, k)
        k_rep = np.repeat(self.k_rot, self.heads // self.kv_heads, axis=2)
        self.v_rep = np.repeat(v, self.heads // self.kv_heads, axis=2)
        self.dots = np.einsum('bshd,bthd->bsht', self.q_rot, k_rep) * self.scale
        self.att = fast_softmax(self.dots)
        out = np.einsum('bsht,bthd->bshd', self.att, self.v_rep)
        return self.wo.forward(out.reshape(b, s, d))

    def backward(self, dout):
        b, s, d = dout.shape
        d_wo_in = self.wo.backward(dout).reshape(b, s, self.heads, self.hd)
        k_rep = np.repeat(self.k_rot, self.heads // self.kv_heads, axis=2)
        d_att = np.einsum('bshd,bthd->bsht', d_wo_in, self.v_rep)
        d_v_rep = np.einsum('bsht,bshd->bthd', self.att, d_wo_in)
        d_dots = self.att * (d_att - np.sum(self.att * d_att, axis=-1, keepdims=True)) * self.scale
        dq_rot = np.einsum('bsht,bthd->bshd', d_dots, k_rep)
        dk_rep = np.einsum('bsht,bshd->bthd', d_dots, self.q_rot)
        dq_rot, dk_rot_rep = self.rope.backward(dq_rot, dk_rep)
        dk_rot = np.sum(dk_rot_rep.reshape(b, s, self.kv_heads, self.heads // self.kv_heads, self.hd), axis=3)
        dv = np.sum(d_v_rep.reshape(b, s, self.kv_heads, self.heads // self.kv_heads, self.hd), axis=3)
        return self.wq.backward(dq_rot.reshape(b, s, d)) + \
               self.wk.backward(dk_rot.reshape(b, s, -1)) + \
               self.wv.backward(dv.reshape(b, s, -1))

class GroqExpert:
    def __init__(self, dim, h_dim):
        self.w1 = Linear(dim, h_dim * 2)
        self.w2 = Linear(h_dim, dim)

    def forward(self, x):
        self.h = self.w1.forward(x)
        self.act = swiglu(self.h)
        return self.w2.forward(self.act)

    def backward(self, dout):
        d_act = self.w2.backward(dout)
        dh = d_swiglu(self.h, d_act)
        return self.w1.backward(dh)

class SovereignMoE:
    def __init__(self, dim, num_experts=4, k=2):
        self.dim, self.num_experts, self.k = dim, num_experts, k
        self.experts = [GroqExpert(dim, dim * 2) for _ in range(num_experts)]
        self.gate = Linear(dim, num_experts)

    def forward(self, x):
        self.orig_shape = x.shape
        x_flat = x.reshape(-1, self.dim)
        logits = self.gate.forward(x_flat)
        probs = fast_softmax(logits)
        idx = np.argsort(probs, axis=-1)[:, -self.k:]
        self.idx, self.probs = idx, probs
        out_flat = np.zeros_like(x_flat)
        self.expert_masks = []
        self.expert_outs = [None] * self.num_experts
        for i in range(self.num_experts):
            mask = np.any(idx == i, axis=1)
            self.expert_masks.append(mask)
            if np.any(mask):
                p = probs[mask, i:i+1]
                e_out = self.experts[i].forward(x_flat[mask])
                self.expert_outs[i] = e_out
                out_flat[mask] += e_out * p
        return out_flat.reshape(self.orig_shape)

    def backward(self, dout):
        dout_flat = dout.reshape(-1, self.dim)
        dx_flat = np.zeros_like(dout_flat)
        dg_logits = np.zeros_like(self.probs)
        for i in range(self.num_experts):
            mask = self.expert_masks[i]
            if np.any(mask):
                p = self.probs[mask, i:i+1]
                dx_flat[mask] += self.experts[i].backward(dout_flat[mask] * p)
                dg_logits[mask, i] = np.sum(dout_flat[mask] * self.expert_outs[i], axis=-1)
        dg = self.probs * (dg_logits - np.sum(self.probs * dg_logits, axis=-1, keepdims=True))
        dx_flat += self.gate.backward(dg)
        return dx_flat.reshape(self.orig_shape)

class SovereignBlock:
    def __init__(self, dim):
        self.norm1 = RMSNorm(dim)
        self.attn = GeminiGQA(dim)
        self.norm2 = RMSNorm(dim)
        self.moe = SovereignMoE(dim)

    def forward(self, x):
        self.n1 = self.norm1.forward(x)
        self.a1 = self.attn.forward(self.n1)
        self.x2 = x + self.a1
        self.n2 = self.norm2.forward(self.x2)
        self.m1 = self.moe.forward(self.n2)
        return self.x2 + self.m1

    def backward(self, dout):
        dm1 = self.moe.backward(dout)
        dx2 = dout + self.norm2.backward(dm1)
        da1 = self.attn.backward(dx2)
        return dx2 + self.norm1.backward(da1)

class OMEGA_ASI_V4:
    def __init__(self, in_dim=784, h_dim=128, out_dim=10, depth=4):
        self.patch_size = 16
        self.num_patches = in_dim // self.patch_size
        self.stem = Linear(self.patch_size, h_dim)
        self.pos_emb = np.random.normal(0, 0.02, (1, self.num_patches, h_dim)).astype(np.float32)
        self.blocks = [SovereignBlock(h_dim) for _ in range(depth)]
        self.norm = RMSNorm(h_dim)
        self.head = Linear(h_dim, out_dim, init_scale=0.1)

    def forward(self, x):
        b = x.shape[0]
        x = x.reshape(b, self.num_patches, self.patch_size)
        x = self.stem.forward(x) + self.pos_emb
        for block in self.blocks: x = block.forward(x)
        self.pooled = np.mean(x, axis=1)
        return self.head.forward(self.norm.forward(self.pooled))

    def backward(self, dout):
        dout = self.norm.backward(self.head.backward(dout))
        dout = np.tile(dout[:, None, :] / self.num_patches, (1, self.num_patches, 1))
        for block in reversed(self.blocks): dout = block.backward(dout)
        self.stem.backward(dout)

    def get_params(self):
        params = []
        def walk(o):
            if isinstance(o, (Linear, RMSNorm)): params.append(o)
            elif hasattr(o, "__dict__"):
                for v in o.__dict__.values():
                    if isinstance(v, list): [walk(i) for i in v]
                    else: walk(v)
        walk(self)
        return params

class Lion:
    def __init__(self, params, lr=1e-4, b1=0.9, b2=0.99, wd=0.01):
        self.params, self.lr, self.b1, self.b2, self.wd = params, lr, b1, b2, wd
        self.m = [np.zeros_like(p.W) if hasattr(p, "W") else np.zeros_like(p.g) for p in params]
        self.m_b = [np.zeros_like(p.b) if hasattr(p, "b") else None for p in params]

    def step(self):
        for i, p in enumerate(self.params):
            if hasattr(p, "W"):
                for a, m_ptr in [("W", self.m), ("b", self.m_b)]:
                    g, m = getattr(p, "d" + a), m_ptr[i]
                    u = np.sign(self.b1 * m + (1.0 - self.b1) * g)
                    w = getattr(p, a)
                    w -= self.lr * (u + self.wd * w if a == "W" else u)
                    m_ptr[i] = self.b2 * m + (1.0 - self.b2) * g
                    setattr(p, a, w)
            else:
                u = np.sign(self.b1 * self.m[i] + (1.0 - self.b1) * p.dg)
                p.g -= self.lr * (u + self.wd * p.g)
                self.m[i] = self.b2 * self.m[i] + (1.0 - self.b2) * p.dg

def get_data(n=2560):
    X = np.random.randn(n, 784).astype(np.float32)
    y = np.random.randint(0, 10, n)
    X += np.eye(10, 784)[y] * 5.0
    return (X - X.mean()) / (X.std() + 1e-6), y

def train():
    X, y = get_data(5120)
    model = OMEGA_ASI_V4(h_dim=64, depth=2)
    params = model.get_params()
    opt = Lion(params, lr=3e-4)
    bs, epochs = 64, 20
    print("OMEGA-ASI V4 | SOVEREIGN-CORE | ONLINE")
    for ep in range(epochs):
        idx = np.random.permutation(len(X))
        l_sum, a_sum, t0 = 0, 0, time.time()
        for i in range(0, len(X), bs):
            xb, yb = X[idx[i:i+bs]], y[idx[i:i+bs]]
            if len(xb) < bs: continue
            probs = fast_softmax(model.forward(xb))
            l_sum += -np.mean(np.log(probs[range(len(yb)), yb] + 1e-10)) * len(yb)
            a_sum += np.sum(np.argmax(probs, axis=1) == yb)
            dout = probs.copy(); dout[range(len(yb)), yb] -= 1
            model.backward(dout / len(yb))
            gn = np.sqrt(sum(np.sum(p.dW**2) + np.sum(p.db**2) for p in params if hasattr(p, "dW")))
            if gn > 1.0:
                for p in params:
                    if hasattr(p, "dW"): p.dW /= gn; p.db /= gn
                    if hasattr(p, "dg"): p.dg /= gn
            opt.step()
        print(f"EP:{ep:02d} | LOSS:{l_sum/len(X):.4f} | ACC:{a_sum/len(X):.4f} | {len(X)/(time.time()-t0):.1f} s/s")

if __name__ == "__main__":
    train()
