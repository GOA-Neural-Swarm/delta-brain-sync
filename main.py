import numpy as np
import time

def fast_softmax(x, axis=-1):
    max_x = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - max_x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)

def swiglu(x):
    # Optimized Swish-Gated Linear Unit approximation
    half = x.shape[-1] // 2
    gate, val = x[..., :half], x[..., half:]
    return (gate * (1.0 / (1.0 + np.exp(-gate)))) * val

def d_swiglu(x, dout):
    half = x.shape[-1] // 2
    gate, val = x[..., :half], x[..., half:]
    sig = 1.0 / (1.0 + np.exp(-gate))
    swish = gate * sig
    d_gate = dout * val * sig * (1.0 + gate * (1.0 - sig))
    d_val = dout * swish
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
        inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2).astype(np.float32) / dim))
        self.inv_freq = inv_freq

    def forward(self, x, seq_len):
        t = np.arange(seq_len, dtype=np.float32)
        freqs = np.outer(t, self.inv_freq)
        emb = np.concatenate((freqs, freqs), axis=-1)
        self.cos = np.cos(emb)[None, :, None, :]
        self.sin = np.sin(emb)[None, :, None, :]
        half = self.dim // 2
        x_rot = np.concatenate((-x[..., half:], x[..., :half]), axis=-1)
        return x * self.cos + x_rot * self.sin

    def backward(self, dout):
        half = self.dim // 2
        d_x_rot = dout * self.sin
        dx = dout * self.cos
        dx[..., :half] += d_x_rot[..., half:]
        dx[..., half:] -= d_x_rot[..., :half]
        return dx

class SovereignAttention:
    def __init__(self, dim, heads=8):
        self.dim, self.heads, self.hd = dim, heads, dim // heads
        self.scale = 1.0 / np.sqrt(self.hd)
        self.wq = Linear(dim, dim)
        self.wk = Linear(dim, dim)
        self.wv = Linear(dim, dim)
        self.wo = Linear(dim, dim)
        self.rope = RotaryEmbedding(self.hd)

    def forward(self, x):
        b, s, d = x.shape
        q = self.wq.forward(x).reshape(b, s, self.heads, self.hd)
        k = self.wk.forward(x).reshape(b, s, self.heads, self.hd)
        v = self.wv.forward(x).reshape(b, s, self.heads, self.hd)
        self.q_rot = self.rope.forward(q, s)
        self.k_rot = self.rope.forward(k, s)
        self.v_t = v.transpose(0, 2, 1, 3)
        
        # Multi-Head Scaled Dot-Product
        self.dots = np.matmul(self.q_rot.transpose(0, 2, 1, 3), self.k_rot.transpose(0, 2, 3, 1)) * self.scale
        self.att = fast_softmax(self.dots)
        out = np.matmul(self.att, self.v_t).transpose(0, 2, 1, 3).reshape(b, s, d)
        return self.wo.forward(out)

    def backward(self, dout):
        b, s, d = dout.shape
        d_wo = self.wo.backward(dout).reshape(b, s, self.heads, self.hd).transpose(0, 2, 1, 3)
        d_att = np.matmul(d_wo, self.v_t.transpose(0, 1, 3, 2))
        d_v_t = np.matmul(self.att.transpose(0, 1, 3, 2), d_wo)
        d_dots = self.att * (d_att - np.sum(self.att * d_att, axis=-1, keepdims=True)) * self.scale
        d_q_t = np.matmul(d_dots, self.k_rot.transpose(0, 2, 1, 3))
        d_k_t = np.matmul(d_dots.transpose(0, 1, 3, 2), self.q_rot.transpose(0, 2, 1, 3))
        d_q = self.rope.backward(d_q_t.transpose(0, 2, 1, 3))
        d_k = self.rope.backward(d_k_t.transpose(0, 2, 1, 3))
        d_v = d_v_t.transpose(0, 2, 1, 3).reshape(b, s, d)
        return self.wq.backward(d_q.reshape(b, s, d)) + self.wk.backward(d_k.reshape(b, s, d)) + self.wv.backward(d_v)

class SovereignMoE:
    def __init__(self, dim, num_experts=4, k=2):
        self.dim, self.num_experts, self.k = dim, num_experts, k
        # SwiGLU experts: Linear(dim, dim*4) -> SwiGLU -> Linear(dim*2, dim)
        self.experts = [[Linear(dim, dim*4), Linear(dim*2, dim)] for _ in range(num_experts)]
        self.gate = Linear(dim, num_experts)

    def forward(self, x):
        self.orig_shape = x.shape
        x_flat = x.reshape(-1, self.dim)
        logits = self.gate.forward(x_flat)
        self.probs = fast_softmax(logits)
        
        # Top-k routing
        self.top_k_indices = np.argsort(self.probs, axis=-1)[:, -self.k:]
        self.top_k_probs = np.take_along_axis(self.probs, self.top_k_indices, axis=-1)
        self.top_k_probs /= (np.sum(self.top_k_probs, axis=-1, keepdims=True) + 1e-12)
        
        out_flat = np.zeros_like(x_flat)
        self.expert_cache = []
        
        for i in range(self.num_experts):
            mask = np.any(self.top_k_indices == i, axis=-1)
            if np.any(mask):
                # Extract tokens for this expert
                idx_in_topk = np.where(self.top_k_indices == i)
                expert_weights = self.top_k_probs[idx_in_topk[0], idx_in_topk[1]][:, None]
                
                e_in = x_flat[mask]
                h_linear = self.experts[i][0].forward(e_in)
                h_swish = swiglu(h_linear)
                e_out = self.experts[i][1].forward(h_swish)
                
                out_flat[mask] += e_out * expert_weights
                self.expert_cache.append((i, mask, h_linear, h_swish, e_out, expert_weights))
            else:
                self.expert_cache.append((i, mask, None, None, None, None))
                
        return out_flat.reshape(self.orig_shape)

    def backward(self, dout):
        dout_flat = dout.reshape(-1, self.dim)
        dx_flat = np.zeros_like(dout_flat)
        dg_logits = np.zeros_like(self.probs)
        
        for i, mask, h_linear, h_swish, e_out, weights in self.expert_cache:
            if np.any(mask):
                de_out = dout_flat[mask] * weights
                dh_swish = self.experts[i][1].backward(de_out)
                dh_linear = d_swiglu(h_linear, dh_swish)
                dx_flat[mask] += self.experts[i][0].backward(dh_linear)
                # Gating gradient
                dg_logits[mask, i] = np.sum(dout_flat[mask] * e_out, axis=-1)
                
        dg = self.probs * (dg_logits - np.sum(self.probs * dg_logits, axis=-1, keepdims=True))
        dx_flat += self.gate.backward(dg)
        return dx_flat.reshape(self.orig_shape)

class SovereignBlockV16:
    def __init__(self, dim):
        self.norm1 = RMSNorm(dim)
        self.attn = SovereignAttention(dim)
        self.norm2 = RMSNorm(dim)
        self.moe = SovereignMoE(dim)
        self.alpha = np.array([1.0], dtype=np.float32)
        self.beta = np.array([1.0], dtype=np.float32)

    def forward(self, x):
        self.x = x
        self.norm1_out = self.norm1.forward(x)
        self.attn_out = self.attn.forward(self.norm1_out)
        self.mid = x + self.alpha * self.attn_out
        self.norm2_out = self.norm2.forward(self.mid)
        self.moe_out = self.moe.forward(self.norm2_out)
        return self.mid + self.beta * self.moe_out

    def backward(self, dout):
        d_moe = self.moe.backward(dout * self.beta)
        d_mid = dout + self.norm2.backward(d_moe)
        d_attn = self.attn.backward(d_mid * self.alpha)
        return d_mid + self.norm1.backward(d_attn)

class OMEGA_ASI_Architect:
    def __init__(self, in_dim=784, h_dim=128, out_dim=10, depth=4):
        self.patch_size = 16
        self.num_patches = in_dim // self.patch_size
        self.stem = Linear(self.patch_size, h_dim)
        self.pos_emb = np.random.normal(0, 0.02, (1, self.num_patches, h_dim)).astype(np.float32)
        self.blocks = [SovereignBlockV16(h_dim) for _ in range(depth)]
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

class LionV2:
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
                    g, m = getattr(p, "d"+a), self.m[i][a]
                    # Lion update: update = sign(b1 * m + (1-b1) * g)
                    u = np.sign(self.b1 * m + (1.0 - self.b1) * g)
                    w = getattr(p, a)
                    w -= self.lr * (u + self.wd * w if a == "W" else u)
                    # Momentum update: m = b2 * m + (1-b2) * g
                    self.m[i][a] = self.b2 * m + (1.0 - self.b2) * g
                    setattr(p, a, w)
            else:
                u = np.sign(self.b1 * self.m[i] + (1.0 - self.b1) * p.dg)
                p.g -= self.lr * u
                self.m[i] = self.b2 * self.m[i] + (1.0 - self.b2) * p.dg

def get_data(n=5000):
    X = np.random.randn(n, 784).astype(np.float32)
    y = np.random.randint(0, 10, n)
    centers = np.random.randn(10, 784).astype(np.float32) * 10.0
    X += centers[y]
    return (X - X.mean()) / (X.std() + 1e-6), y

def train():
    X, y = get_data(6000)
    model = OMEGA_ASI_Architect(h_dim=64, depth=3)
    params = model.get_params()
    lr_init = 3e-4
    opt = LionV2(params, lr=lr_init, wd=0.02)
    bs, epochs = 64, 30
    
    print("OMEGA-ASI | V16-SOVEREIGN-EVOLUTION | RECURSIVE SELF-EVOLUTION START")
    
    for ep in range(epochs):
        idx = np.random.permutation(len(X))
        l_sum, a_sum, t0 = 0, 0, time.time()
        
        # Cosine Learning Rate Decay
        opt.lr = lr_init * 0.5 * (1 + np.cos(np.pi * ep / epochs))
        
        for i in range(0, len(X), bs):
            xb, yb = X[idx[i:i+bs]], y[idx[i:i+bs]]
            
            # Forward
            logits = model.forward(xb)
            probs = fast_softmax(logits)
            
            # Loss (Cross-Entropy)
            batch_indices = np.arange(len(yb))
            loss = -np.mean(np.log(probs[batch_indices, yb] + 1e-12))
            l_sum += loss * len(yb)
            a_sum += np.sum(np.argmax(probs, axis=1) == yb)
            
            # Backward
            dout = probs.copy()
            dout[batch_indices, yb] -= 1
            model.backward(dout / len(yb))
            
            # Global Gradient Clipping
            gn = np.sqrt(sum(np.sum(p.dW**2) + np.sum(p.db**2) for p in params if hasattr(p, "dW")))
            if gn > 1.0:
                for p in params:
                    if hasattr(p, "dW"): p.dW /= gn; p.db /= gn
                    if hasattr(p, "dg"): p.dg /= gn
            
            opt.step()
            
        dt = time.time() - t0
        print(f"EP:{ep:02d} | LOSS:{l_sum/len(X):.4f} | ACC:{a_sum/len(X):.4f} | LR:{opt.lr:.6f} | {len(X)/dt:.0f} samples/s")

if __name__ == "__main__":
    train()
