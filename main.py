import numpy as np

class Kernels:
    @staticmethod
    def softmax(x, axis=-1):
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)

    @staticmethod
    def swiglu_f(x, w, v):
        # Gemini-optimized activation path
        gate = x @ w
        res = gate * (1 / (1 + np.exp(-np.clip(gate, -12, 12))))
        return res * (x @ v)

    @staticmethod
    def swiglu_b(x, w, v, dy):
        # High-performance gradient reconstruction
        g = x @ w
        v_p = x @ v
        sig = 1 / (1 + np.exp(-np.clip(g, -12, 12)))
        swish = g * sig
        ds = dy * v_p
        dg = ds * (sig + swish * (1 - sig))
        dv = dy * swish
        dx = dg @ w.T + dv @ v.T
        dw = x.T @ dg
        dv_w = x.T @ dv
        return dx, dw, dv_w

class Linear:
    def __init__(self, i, o, name=""):
        self.W = (np.random.randn(i, o) * np.sqrt(2 / i)).astype(np.float32)
        self.b = np.zeros(o, dtype=np.float32)
        self.dW, self.db = None, None
        self.name = name

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dy):
        self.dW = self.x.reshape(-1, self.x.shape[-1]).T @ dy.reshape(-1, dy.shape[-1])
        self.db = np.sum(dy, axis=tuple(range(dy.ndim - 1)))
        return dy @ self.W.T

class RMSNorm:
    def __init__(self, d, eps=1e-6):
        self.g = np.ones(d, dtype=np.float32)
        self.eps = eps

    def forward(self, x):
        self.x = x
        self.rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return self.g * (x / self.rms)

    def backward(self, dy):
        x_rms = self.x / self.rms
        self.dg = np.sum(dy * x_rms, axis=tuple(range(dy.ndim - 1)))
        d_norm = dy * self.g
        return (1 / self.rms) * (d_norm - x_rms * np.mean(d_norm * x_rms, axis=-1, keepdims=True))

class RotaryPositionalEmbedding:
    def __init__(self, d, max_seq=2048):
        inv_freq = 1.0 / (10000 ** (np.arange(0, d, 2).astype(np.float32) / d))
        t = np.arange(max_seq, dtype=np.float32)
        freqs = np.outer(t, inv_freq)
        self.cos = np.cos(freqs)
        self.sin = np.sin(freqs)

    def apply(self, x, conj=False):
        b, s, h, d = x.shape
        x1, x2 = x[..., :d//2], x[..., d//2:]
        c, sn = self.cos[:s, None, :], self.sin[:s, None, :]
        if conj:
            return np.concatenate([x1 * c + x2 * sn, x2 * c - x1 * sn], axis=-1)
        return np.concatenate([x1 * c - x2 * sn, x2 * c + x1 * sn], axis=-1)

class GQA:
    def __init__(self, d, heads=8, groups=2):
        self.d, self.h, self.g = d, heads, groups
        self.hd = d // heads
        self.wq = Linear(d, d)
        self.wk = Linear(d, (heads // groups) * self.hd)
        self.wv = Linear(d, (heads // groups) * self.hd)
        self.wo = Linear(d, d)
        self.rope = RotaryPositionalEmbedding(self.hd)
        self.scale = self.hd ** -0.5

    def forward(self, x):
        b, s, _ = x.shape
        q = self.wq.forward(x).reshape(b, s, self.h, self.hd)
        k = self.wk.forward(x).reshape(b, s, self.h // self.g, self.hd)
        v = self.wv.forward(x).reshape(b, s, self.h // self.g, self.hd)
        
        self.q_rope = self.rope.apply(q)
        self.k_rope = self.rope.apply(k)
        self.v_val = v
        
        # Expand KV for GQA
        k_ext = np.repeat(self.k_rope, self.g, axis=2)
        v_ext = np.repeat(self.v_val, self.g, axis=2)
        
        attn_scores = np.einsum("bshd,bthd->bsht", self.q_rope, k_ext) * self.scale
        self.probs = Kernels.softmax(attn_scores)
        
        out = np.einsum("bsht,bthd->bshd", self.probs, v_ext)
        return self.wo.forward(out.reshape(b, s, self.d))

    def backward(self, dy):
        b, s, _ = dy.shape
        dy_wo = self.wo.backward(dy).reshape(b, s, self.h, self.hd)
        
        k_ext = np.repeat(self.k_rope, self.g, axis=2)
        v_ext = np.repeat(self.v_val, self.g, axis=2)
        
        d_probs = np.einsum("bshd,bthd->bsht", dy_wo, v_ext)
        d_attn = self.probs * (d_probs - np.sum(self.probs * d_probs, axis=-1, keepdims=True)) * self.scale
        
        dq = self.rope.apply(np.einsum("bsht,bthd->bshd", d_attn, k_ext), conj=True)
        dk_ext = np.einsum("bsht,bshd->bthd", d_attn, self.q_rope)
        dv_ext = np.einsum("bsht,bshd->bthd", self.probs, dy_wo)
        
        dk = self.rope.apply(dk_ext.reshape(b, s, self.h // self.g, self.g, self.hd).sum(3), conj=True)
        dv = dv_ext.reshape(b, s, self.h // self.g, self.g, self.hd).sum(3)
        
        return self.wq.backward(dq.reshape(b, s, -1)) + \
               self.wk.backward(dk.reshape(b, s, -1)) + \
               self.wv.backward(dv.reshape(b, s, -1))

class GeminiGroqDualPath:
    def __init__(self, d, expand=4):
        # Gemini Path: High-Capacity SwiGLU
        self.w1 = Linear(d, d * expand)
        self.v1 = Linear(d, d * expand)
        self.w2 = Linear(d * expand, d)
        # Groq Path: High-Throughput Linear
        self.w_groq = Linear(d, d)
        # Sovereign Gating
        self.gate = Linear(d, 2)

    def forward(self, x):
        self.x = x
        self.g_out = Kernels.swiglu_f(x, self.w1.W, self.v1.W) @ self.w2.W
        self.q_out = self.w_groq.forward(x)
        self.routing = Kernels.softmax(self.gate.forward(x))
        return self.routing[..., 0:1] * self.g_out + self.routing[..., 1:2] * self.q_out

    def backward(self, dy):
        r0, r1 = self.routing[..., 0:1], self.routing[..., 1:2]
        
        # Gating Gradients
        dr0 = np.sum(dy * self.g_out, axis=-1, keepdims=True)
        dr1 = np.sum(dy * self.q_out, axis=-1, keepdims=True)
        dr = np.concatenate([dr0, dr1], axis=-1)
        d_gate = self.gate.backward(self.routing * (dr - np.sum(self.routing * dr, axis=-1, keepdims=True)))
        
        # Path Gradients
        dy_g = dy * r0
        dy_q = dy * r1
        
        # Groq Backward
        dx_q = self.w_groq.backward(dy_q)
        
        # Gemini Backward
        dy_w2 = (Kernels.swiglu_f(self.x, self.w1.W, self.v1.W)).T @ dy_g.reshape(-1, dy_g.shape[-1])
        self.w2.dW = dy_w2
        dy_swi = dy_g @ self.w2.W.T
        dx_g, dw1, dv1 = Kernels.swiglu_b(self.x, self.w1.W, self.v1.W, dy_swi)
        self.w1.dW, self.v1.dW = dw1, dv1
        
        return dx_q + dx_g + d_gate

class SovereignBlock:
    def __init__(self, d):
        self.ln1 = RMSNorm(d)
        self.attn = GQA(d)
        self.ln2 = RMSNorm(d)
        self.dual_path = GeminiGroqDualPath(d)

    def forward(self, x):
        x = x + self.attn.forward(self.ln1.forward(x))
        x = x + self.dual_path.forward(self.ln2.forward(x))
        return x

    def backward(self, dy):
        dy_path = self.dual_path.backward(dy)
        dy = dy + self.ln2.backward(dy_path)
        dy_attn = self.attn.backward(dy)
        dy = dy + self.ln1.backward(dy_attn)
        return dy

class OMEGA_ASI_Model:
    def __init__(self, dims, h_dim, out_dim, depth=4):
        self.embed = Linear(dims, h_dim)
        self.blocks = [SovereignBlock(h_dim) for _ in range(depth)]
        self.final_ln = RMSNorm(h_dim)
        self.head = Linear(h_dim, out_dim)

    def forward(self, x):
        if x.ndim == 2: x = x[:, None, :]
        x = self.embed.forward(x)
        for b in self.blocks: x = b.forward(x)
        return self.head.forward(self.final_ln.forward(x[:, -1, :]))

    def backward(self, dy):
        dy = self.final_ln.backward(self.head.backward(dy))
        dy_seq = np.zeros((dy.shape[0], 1, dy.shape[1]), dtype=np.float32)
        dy_seq[:, -1, :] = dy
        for b in reversed(self.blocks): dy_seq = b.backward(dy_seq)
        self.embed.backward(dy_seq)

    def get_params(self):
        params = []
        def find(obj):
            if isinstance(obj, (Linear, RMSNorm)): params.append(obj)
            elif isinstance(obj, list): [find(i) for i in obj]
            elif hasattr(obj, "__dict__"): [find(v) for k, v in obj.__dict__.items() if k not in ('x', 'rms', 'probs', 'routing', 'g_out', 'q_out', 'q_rope', 'k_rope', 'v_val')]
        find(self)
        return list(set(params))

class LionOptimizer:
    def __init__(self, params, lr=1e-4, b1=0.9, b2=0.99, wd=0.01):
        self.params = params
        self.lr, self.b1, self.b2, self.wd = lr, b1, b2, wd
        self.m = {id(p): [np.zeros_like(getattr(p, a)) for a in (["W", "b"] if hasattr(p, "W") else ["g"])] for p in params}

    def step(self, scale=1.0):
        lr = self.lr * scale
        for p in self.params:
            attrs = ["W", "b"] if hasattr(p, "W") else ["g"]
            for i, a in enumerate(attrs):
                grad = getattr(p, "d" + a if a != "g" else "dg")
                val = getattr(p, a)
                m = self.m[id(p)][i]
                
                update = np.sign(self.b1 * m + (1 - self.b1) * grad)
                val -= lr * (update + self.wd * val if a in ("W", "g") else update)
                self.m[id(p)][i] = self.b2 * m + (1 - self.b2) * grad
                setattr(p, a, val)

def train():
    # Synthetic Data: 784 features (MNIST-like)
    N, D, C, BS, EPOCHS = 2048, 784, 10, 64, 40
    X = (np.random.randn(N, D) * 0.01).astype(np.float32)
    Y = np.random.randint(0, C, N)
    
    model = OMEGA_ASI_Model(D, 128, C, depth=3)
    params = model.get_params()
    optimizer = LionOptimizer(params, lr=2e-4, wd=0.05)
    
    for epoch in range(EPOCHS):
        indices = np.random.permutation(N)
        total_loss, total_acc = 0, 0
        lr_scale = 0.5 * (1 + np.cos(np.pi * epoch / EPOCHS))
        
        for i in range(0, N, BS):
            batch_idx = indices[i:i+BS]
            xb, yb = X[batch_idx], Y[batch_idx]
            
            logits = model.forward(xb)
            probs = Kernels.softmax(logits)
            
            # Cross Entropy Loss
            loss = -np.mean(np.log(probs[range(len(yb)), yb] + 1e-12))
            total_loss += loss * len(yb)
            total_acc += np.sum(np.argmax(probs, axis=1) == yb)
            
            # Backward
            dy = probs.copy()
            dy[range(len(yb)), yb] -= 1
            model.backward(dy / len(yb))
            
            # Global Gradient Clipping
            gnorm = np.sqrt(sum(np.sum(getattr(p, "d"+a if a!="g" else "dg")**2) for p in params for a in (["W", "b"] if hasattr(p, "W") else ["g"])))
            if gnorm > 1.0:
                for p in params:
                    for a in (["W", "b"] if hasattr(p, "W") else ["g"]):
                        gn = "d"+a if a!="g" else "dg"
                        setattr(p, gn, getattr(p, gn) / gnorm)
            
            optimizer.step(lr_scale)
            
        print(f"Epoch {epoch+1:02d} | Loss: {total_loss/N:.4f} | Acc: {total_acc/N:.4f} | LR: {optimizer.lr*lr_scale:.6f}")

if __name__ == "__main__":
    train()
