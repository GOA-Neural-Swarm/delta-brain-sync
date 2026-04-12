
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
    def __init__(self, dim, max_seq_len=128):
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2).astype(np.float32) / dim))
        t = np.arange(max_seq_len, dtype=np.float32)
        freqs = np.outer(t, inv_freq)
        emb = np.concatenate((freqs, freqs), axis=-1)
        self.cos_cache = np.cos(emb)[None, :, None, :]
        self.sin_cache = np.sin(emb)[None, :, None, :]

    def forward(self, q, k):
        s = q.shape[1]
        cos, sin = self.cos_cache[:, :s, :, :], self.sin_cache[:, :s, :, :]
        self.cos, self.sin = cos, sin
        def rotate(x):
            x_rot = np.concatenate((-x[..., self.dim//2:], x[..., :self.dim//2]), axis=-1)
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

class GeminiAttention:
    def __init__(self, dim, heads=8):
        self.dim, self.heads, self.hd = dim, heads, dim // heads
        self.scale = 1.0 / np.sqrt(self.hd)
        self.wq, self.wk, self.wv, self.wo = [Linear(dim, dim) for _ in range(4)]
        self.rope = RotaryEmbedding(self.hd)

    def forward(self, x):
        b, s, d = x.shape
        q = self.wq.forward(x).reshape(b, s, self.heads, self.hd)
        k = self.wk.forward(x).reshape(b, s, self.heads, self.hd)
        v = self.wv.forward(x).reshape(b, s, self.heads, self.hd)
        self.q_rot, self.k_rot = self.rope.forward(q, k)
        self.v_t = v.transpose(0, 2, 1, 3)
        self.dots = np.matmul(self.q_rot.transpose(0, 2, 1, 3), self.k_rot.transpose(0, 2, 3, 1)) * self.scale
        self.att = fast_softmax(self.dots)
        out = np.matmul(self.att, self.v_t).transpose(0, 2, 1, 3).reshape(b, s, d)
        return self.wo.forward(out)

    def backward(self, dout):
        b, s, d = dout.shape
        d_wo_in = self.wo.backward(dout).reshape(b, s, self.heads, self.hd).transpose(0, 2, 1, 3)
        d_att = np.matmul(d_wo_in, self.v_t.transpose(0, 1, 3, 2))
        d_v_t = np.matmul(self.att.transpose(0, 1, 3, 2), d_wo_in)
        d_dots = self.att * (d_att - np.sum(self.att * d_att, axis=-1, keepdims=True)) * self.scale
        dq_rot_t = np.matmul(d_dots, self.k_rot.transpose(0, 2, 1, 3))
        dk_rot_t = np.matmul(d_dots.transpose(0, 1, 3, 2), self.q_rot.transpose(0, 2, 1, 3))
        dq_rot, dk_rot = dq_rot_t.transpose(0, 2, 1, 3), dk_rot_t.transpose(0, 2, 1, 3)
        dq, dk = self.rope.backward(dq_rot, dk_rot)
        dv = d_v_t.transpose(0, 2, 1, 3)
        return self.wq.backward(dq.reshape(b, s, d)) + self.wk.backward(dk.reshape(b, s, d)) + self.wv.backward(dv.reshape(b, s, d))

class GroqExpert:
    def __init__(self, dim):
        self.w1 = Linear(dim, dim * 4)
        self.w2 = Linear(dim * 2, dim)

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
        self.experts = [GroqExpert(dim) for _ in range(num_experts)]
        self.gate = Linear(dim, num_experts)

    def forward(self, x):
        self.orig_shape = x.shape
        x_flat = x.reshape(-1, self.dim)
        logits = self.gate.forward(x_flat)
        probs = fast_softmax(logits)
        top_k_indices = np.argsort(probs, axis=-1)[:, -self.k:]
        self.top_k_indices = top_k_indices
        rows = np.arange(x_flat.shape[0])[:, None]
        top_k_probs = probs[rows, top_k_indices]
        top_k_probs /= (np.sum(top_k_probs, axis=-1, keepdims=True) + 1e-12)
        self.top_k_probs = top_k_probs
        out_flat = np.zeros_like(x_flat)
        self.expert_masks = []
        self.expert_outs = [None] * self.num_experts
        for i in range(self.num_experts):
            mask = np.any(top_k_indices == i, axis=1)
            self.expert_masks.append(mask)
            if np.any(mask):
                slot_idx = np.where(top_k_indices[mask] == i)[1]
                p = top_k_probs[mask, slot_idx][:, None]
                e_out = self.experts[i].forward(x_flat[mask])
                self.expert_outs[i] = e_out
                out_flat[mask] += e_out * p
        self.probs = probs
        return out_flat.reshape(self.orig_shape)

    def backward(self, dout):
        dout_flat = dout.reshape(-1, self.dim)
        x_flat = self.gate.x
        dx_flat = np.zeros_like(dout_flat)
        dg_logits = np.zeros_like(self.probs)
        for i in range(self.num_experts):
            mask = self.expert_masks[i]
            if np.any(mask):
                slot_idx = np.where(self.top_k_indices[mask] == i)[1]
                p = self.top_k_probs[mask, slot_idx][:, None]
                dx_flat[mask] += self.experts[i].backward(dout_flat[mask] * p)
                dg_logits[mask, i] = np.sum(dout_flat[mask] * self.expert_outs[i], axis=-1)
        dg = self.probs * (dg_logits - np.sum(self.probs * dg_logits, axis=-1, keepdims=True))
        dx_flat += self.gate.backward(dg)
        return dx_flat.reshape(self.orig_shape)

class SovereignBlock:
    def __init__(self, dim):
        self.norm1 = RMSNorm(dim)
        self.attn = GeminiAttention(dim)
        self.norm2 = RMSNorm(dim)
        self.moe = SovereignMoE(dim)

    def forward(self, x):
        self.n1 = self.norm1.forward(x)
        self.a1 = self.attn.forward(self.n1)
        self.mid = x + self.a1
        self.n2 = self.norm2.forward(self.mid)
        self.m1 = self.moe.forward(self.n2)
        return self.mid + self.m1

    def backward(self, dout):
        dm1 = self.moe.backward(dout)
        dmid = dout + self.norm2.backward(dm1)
        da1 = self.attn.backward(dmid)
        return dmid + self.norm1.backward(da1)

class OMEGA_ASI_V18:
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
        for block in self.blocks:
            x = block.forward(x)
        self.pooled = np.mean(x, axis=1)
        return self.head.forward(self.norm.forward(self.pooled))

    def backward(self, dout):
        dout = self.norm.backward(self.head.backward(dout))
        dout = np.tile(dout[:, None, :] / self.num_patches, (1, self.num_patches, 1))
        for block in reversed(self.blocks):
            dout = block.backward(dout)
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

class LionOptimizer:
    def __init__(self, params, lr=1e-4, b1=0.9, b2=0.99, wd=0.01):
        self.params, self.lr, self.b1, self.b2, self.wd = params, lr, b1, b2, wd
        self.m = []
        for p in params:
            if hasattr(p, "W"): self.m.append({"W": np.zeros_like(p.W), "b": np.zeros_like(p.b)})
            else: self.m.append(np.zeros_like(p.g))

    def step(self):
        for i, p in enumerate(self.params):
            if hasattr(p, "W"):
                for a in ["W", "b"]:
                    g, m = getattr(p, "d" + a), self.m[i][a]
                    u = np.sign(self.b1 * m + (1.0 - self.b1) * g)
                    w = getattr(p, a)
                    w -= self.lr * (u + self.wd * w if a == "W" else u)
                    self.m[i][a] = self.b2 * m + (1.0 - self.b2) * g
                    setattr(p, a, w)
            else:
                u = np.sign(self.b1 * self.m[i] + (1.0 - self.b1) * p.dg)
                p.g -= self.lr * (u + self.wd * p.g)
                self.m[i] = self.b2 * self.m[i] + (1.0 - self.b2) * p.dg

def get_data(n=5000):
    X = np.random.randn(n, 784).astype(np.float32)
    y = np.random.randint(0, 10, n)
    centers = np.random.randn(10, 784).astype(np.float32) * 10.0
    X += centers[y]
    return (X - X.mean()) / (X.std() + 1e-6), y

def train():
    X, y = get_data(5000)
    model = OMEGA_ASI_V18(h_dim=64, depth=4)
    params = model.get_params()
    lr_init = 1e-3
    opt = LionOptimizer(params, lr=lr_init, wd=0.01)
    bs, epochs = 64, 40
    print("OMEGA-ASI | V18-RECURSIVE-EVOLUTION | ONLINE")
    for ep in range(epochs):
        idx = np.random.permutation(len(X))
        l_sum, a_sum, t0 = 0, 0, time.time()
        opt.lr = lr_init * 0.5 * (1 + np.cos(np.pi * ep / epochs))
        for i in range(0, len(X), bs):
            xb, yb = X[idx[i : i + bs]], y[idx[i : i + bs]]
            if len(xb) < bs: continue
            logits = model.forward(xb)
            probs = fast_softmax(logits)
            loss = -np.mean(np.log(probs[range(len(yb)), yb] + 1e-10))
            l_sum += loss * len(yb)
            a_sum += np.sum(np.argmax(probs, axis=1) == yb)
            dout = (probs.copy())
            dout[range(len(yb)), yb] -= 1
            model.backward(dout / len(yb))
            gn = np.sqrt(sum(np.sum(p.dW**2) + np.sum(p.db**2) for p in params if hasattr(p, "dW")))
            if gn > 1.0:
                for p in params:
                    if hasattr(p, "dW"): p.dW /= gn; p.db /= gn
                    if hasattr(p, "dg"): p.dg /= gn
            opt.step()
        dt = time.time() - t0
        print(f"STEP:{ep:03d} | LOSS:{l_sum/len(X):.4f} | ACC:{a_sum/len(X):.4f} | SPEED:{len(X)/dt:.1f} samples/s")

if __name__ == "__main__":
    train()
