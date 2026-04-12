import numpy as np
import time

def fast_softmax(x, axis=-1):
    max_x = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - max_x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)

def swiglu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))

def d_swiglu(x):
    s = 1.0 / (1.0 + np.exp(-x))
    return s * (1.0 + x * (1.0 - s))

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
        self.x, self.rstd = None, None

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
    def __init__(self, dim):
        self.dim = dim
        self.inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2).astype(np.float32) / dim))

    def forward(self, x, seq_len):
        t = np.arange(seq_len, dtype=np.float32)
        freqs = np.outer(t, self.inv_freq)
        emb = np.concatenate((freqs, freqs), axis=-1)
        cos, sin = np.cos(emb), np.sin(emb)
        x_rot = np.concatenate((-x[..., self.dim//2:], x[..., :self.dim//2]), axis=-1)
        return x * cos + x_rot * sin

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
        
        self.q = np.array([self.rope.forward(q[i], s) for i in range(b)]).transpose(0, 2, 1, 3)
        self.k = np.array([self.rope.forward(k[i], s) for i in range(b)]).transpose(0, 2, 1, 3)
        self.v = v.transpose(0, 2, 1, 3)

        self.dots = np.matmul(self.q, self.k.transpose(0, 1, 3, 2)) * self.scale
        self.att = fast_softmax(self.dots)
        out = np.matmul(self.att, self.v).transpose(0, 2, 1, 3).reshape(b, s, d)
        return self.wo.forward(out)

    def backward(self, dout):
        b, s, d = dout.shape
        d_wo = self.wo.backward(dout).reshape(b, s, self.heads, self.hd).transpose(0, 2, 1, 3)
        d_att = np.matmul(d_wo, self.v.transpose(0, 1, 3, 2))
        d_v = np.matmul(self.att.transpose(0, 1, 3, 2), d_wo)
        d_dots = self.att * (d_att - np.sum(self.att * d_att, axis=-1, keepdims=True)) * self.scale
        d_q = np.matmul(d_dots, self.k).transpose(0, 2, 1, 3).reshape(b, s, d)
        d_k = np.matmul(d_dots.transpose(0, 1, 3, 2), self.q).transpose(0, 2, 1, 3).reshape(b, s, d)
        d_v = d_v.transpose(0, 2, 1, 3).reshape(b, s, d)
        return self.wq.backward(d_q) + self.wk.backward(d_k) + self.wv.backward(d_v)

class GroqMoE:
    def __init__(self, dim, num_experts=8, k=2):
        self.num_experts, self.k = num_experts, k
        self.experts = [Expert(dim) for _ in range(num_experts)]
        self.gate = Linear(dim, num_experts)

    def forward(self, x):
        self.x_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        logits = self.gate.forward(x_flat)
        probs = fast_softmax(logits)
        
        self.top_k_indices = np.argsort(probs, axis=-1)[:, -self.k:]
        self.top_k_probs = np.take_along_axis(probs, self.top_k_indices, axis=-1)
        self.top_k_probs /= (np.sum(self.top_k_probs, axis=-1, keepdims=True) + 1e-12)
        
        out_flat = np.zeros_like(x_flat)
        self.expert_masks = []
        self.expert_outs = []
        
        for i in range(self.num_experts):
            mask = np.any(self.top_k_indices == i, axis=-1)
            self.expert_masks.append(mask)
            if np.any(mask):
                e_out = self.experts[i].forward(x_flat[mask])
                self.expert_outs.append(e_out)
                idx = np.where(self.top_k_indices == i)
                w = self.top_k_probs[idx[0], idx[1]]
                out_flat[mask] += e_out * w[:, np.newaxis]
            else:
                self.expert_outs.append(None)
        
        self.probs = probs
        return out_flat.reshape(self.x_shape)

    def backward(self, dout):
        dout_flat = dout.reshape(-1, dout.shape[-1])
        dx_flat = np.zeros_like(dout_flat)
        dg_logits = np.zeros_like(self.probs)
        
        for i in range(self.num_experts):
            mask = self.expert_masks[i]
            if np.any(mask):
                idx = np.where(self.top_k_indices == i)
                w = self.top_k_probs[idx[0], idx[1]]
                de_out = dout_flat[mask] * w[:, np.newaxis]
                dx_flat[mask] += self.experts[i].backward(de_out)
                dg_logits[mask, i] = np.sum(dout_flat[mask] * self.expert_outs[i], axis=-1)
        
        dg = self.probs * (dg_logits - np.sum(self.probs * dg_logits, axis=-1, keepdims=True))
        dx_flat += self.gate.backward(dg)
        return dx_flat.reshape(self.x_shape)

class Expert:
    def __init__(self, dim):
        self.w1 = Linear(dim, dim * 2)
        self.w2 = Linear(dim * 2, dim)

    def forward(self, x):
        self.x1 = self.w1.forward(x)
        self.act = swiglu(self.x1)
        return self.w2.forward(self.act)

    def backward(self, dout):
        dx1 = self.w2.backward(dout) * d_swiglu(self.x1)
        return self.w1.backward(dx1)

class SovereignBlockV13:
    def __init__(self, dim):
        self.ln1 = RMSNorm(dim)
        self.attn = GeminiAttention(dim)
        self.ln2 = RMSNorm(dim)
        self.moe = GroqMoE(dim)
        self.alpha = np.ones(1, dtype=np.float32) * 0.5

    def forward(self, x):
        self.x = x
        self.attn_res = self.attn.forward(self.ln1.forward(x))
        self.mid = x + self.attn_res
        self.moe_res = self.moe.forward(self.ln2.forward(self.mid))
        return self.mid + self.moe_res

    def backward(self, dout):
        d_moe = self.moe.backward(dout)
        d_mid = dout + self.ln2.backward(d_moe)
        d_attn = self.attn.backward(d_mid)
        return d_mid + self.ln1.backward(d_attn)

class SovereignArchitectV13:
    def __init__(self, h_d=128, out_d=10, depth=4):
        self.patch_dim, self.num_patches = 16, 49
        self.stem = Linear(self.patch_dim, h_d)
        self.pos_emb = np.random.randn(1, self.num_patches, h_d).astype(np.float32) * 0.02
        self.blocks = [SovereignBlockV13(h_d) for _ in range(depth)]
        self.norm = RMSNorm(h_d)
        self.head = Linear(h_d, out_d, init_scale=0.1)

    def forward(self, x):
        b = x.shape[0]
        x = x.reshape(b, self.num_patches, self.patch_dim)
        x = self.stem.forward(x) + self.pos_emb
        for block in self.blocks:
            x = block.forward(x)
        self.pooled = np.mean(x, axis=1)
        return self.head.forward(self.norm.forward(self.pooled))

    def backward(self, dout):
        dout = self.norm.backward(self.head.backward(dout))
        dout = np.tile(dout[:, np.newaxis, :] / self.num_patches, (1, self.num_patches, 1))
        for block in reversed(self.blocks):
            dout = block.backward(dout)
        self.stem.backward(dout)

    def get_params(self):
        params = []
        def walk(obj):
            if isinstance(obj, (Linear, RMSNorm)): params.append(obj)
            elif hasattr(obj, "__dict__"):
                for v in obj.__dict__.values():
                    if isinstance(v, list): [walk(i) for i in v]
                    else: walk(v)
        walk(self)
        return params

class Lion:
    def __init__(self, params, lr=1e-4, b1=0.9, b2=0.99, wd=0.01):
        self.params, self.lr, self.b1, self.b2, self.wd = params, lr, b1, b2, wd
        self.m = []
        for p in params:
            if hasattr(p, "W"): self.m.append({"W": np.zeros_like(p.W), "b": np.zeros_like(p.b)})
            else: self.m.append(np.zeros_like(p.g))

    def step(self):
        for i, p in enumerate(self.params):
            if hasattr(p, "W"):
                for attr in ["W", "b"]:
                    g = getattr(p, "d" + attr)
                    m = self.m[i][attr]
                    u = np.sign(self.b1 * m + (1.0 - self.b1) * g)
                    w = getattr(p, attr)
                    w -= self.lr * (u + self.wd * w if attr == "W" else u)
                    self.m[i][attr] = self.b2 * m + (1.0 - self.b2) * g
                    setattr(p, attr, w)
            else:
                u = np.sign(self.b1 * self.m[i] + (1.0 - self.b1) * p.dg)
                p.g -= self.lr * u
                self.m[i] = self.b2 * self.m[i] + (1.0 - self.b2) * p.dg

def generate_data(n=2000):
    X = np.random.randn(n, 784).astype(np.float32)
    y = np.random.randint(0, 10, n)
    centers = np.random.randn(10, 784).astype(np.float32) * 5.0
    X += centers[y]
    return (X - np.mean(X)) / (np.std(X) + 1e-6), y

def train():
    X, y = generate_data(5000)
    model = SovereignArchitectV13(h_d=64, out_d=10, depth=2)
    params = model.get_params()
    opt = Lion(params, lr=2e-4, wd=0.01)
    bs, epochs = 32, 20

    print("OMEGA-ASI | V13-SOVEREIGN-CORE | BIMODAL REDUNDANCY ACTIVE")

    for ep in range(epochs):
        idx = np.random.permutation(len(X))
        l_acc, a_acc, t0 = 0, 0, time.time()
        for i in range(0, len(X), bs):
            bi = idx[i : i + bs]
            xb, yb = X[bi], y[bi]
            logits = model.forward(xb)
            probs = fast_softmax(logits)
            loss = -np.mean(np.log(probs[range(len(yb)), yb] + 1e-10))
            l_acc += loss * (len(yb) / len(X))
            a_acc += np.mean(np.argmax(probs, axis=1) == yb) * (len(yb) / len(X))
            dout = probs.copy()
            dout[range(len(yb)), yb] -= 1
            model.backward(dout / len(yb))
            
            gn = np.sqrt(sum(np.sum(p.dW**2) + np.sum(p.db**2) for p in params if hasattr(p, "dW")))
            if gn > 1.0:
                for p in params:
                    if hasattr(p, "dW"): p.dW /= gn; p.db /= gn
                    if hasattr(p, "dg"): p.dg /= gn
            opt.step()
        
        print(f"EP:{ep:02d} | LOSS:{l_acc:.4f} | ACC:{a_acc:.4f} | {len(X)/(time.time()-t0):.0f} s/s")

if __name__ == "__main__":
    train()
