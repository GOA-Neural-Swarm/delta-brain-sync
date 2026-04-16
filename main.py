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
        self.name = name

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dy):
        self.dW = self.x.reshape(-1, self.x.shape[-1]).T @ dy.reshape(-1, dy.shape[-1])
        self.db = dy.reshape(-1, dy.shape[-1]).sum(axis=0)
        return (dy @ self.W.T).reshape(self.x.shape)

class RMSNorm:
    def __init__(self, d, eps=1e-6):
        self.g, self.eps = np.ones(d, "f4"), eps

    def forward(self, x):
        self.x = x
        self.rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        self.xn = x / self.rms
        return self.g * self.xn

    def backward(self, dy):
        dn = dy * self.g
        self.dg = np.sum(dy * self.xn, axis=tuple(range(dy.ndim - 1)))
        return (dn - self.xn * np.mean(dn * self.xn, axis=-1, keepdims=True)) / self.rms

class RoPE:
    def __init__(self, d, m=4096):
        f = 10000.0**-(np.arange(0, d, 2) / d)
        t = np.outer(np.arange(m), f)
        self.cos, self.sin = np.cos(t), np.sin(t)

    def apply(self, x, conj=False):
        s, d = x.shape[1], x.shape[-1]
        x1, x2 = x[..., :d//2], x[..., d//2:]
        c, sn = self.cos[:s, None, :], self.sin[:s, None, :]
        if conj: return np.concatenate([x1*c+x2*sn, x2*c-x1*sn], -1)
        return np.concatenate([x1*c-x2*sn, x2*c+x1*sn], -1)

class GQA:
    def __init__(self, d, h=8, g=2):
        self.d, self.h, self.g, self.hd = d, h, g, d // h
        self.wq, self.wk, self.wv, self.wo = Linear(d, d), Linear(d, d//g), Linear(d, d//g), Linear(d, d)
        self.rope, self.scale = RoPE(self.hd), (d//h)**-0.5

    def forward(self, x):
        b, s, _ = x.shape
        q = self.wq.forward(x).reshape(b, s, self.h, self.hd)
        k = self.wk.forward(x).reshape(b, s, self.h//self.g, self.hd)
        v = self.wv.forward(x).reshape(b, s, self.h//self.g, self.hd)
        self.qr, self.kr, self.v_raw = self.rope.apply(q), self.rope.apply(k), v
        ke, ve = np.repeat(self.kr, self.g, 2), np.repeat(v, self.g, 2)
        attn = np.einsum("bshd,bthd->bsht", self.qr, ke) * self.scale
        attn -= np.max(attn, -1, keepdims=True)
        self.p = np.exp(attn); self.p /= (np.sum(self.p, -1, keepdims=True) + 1e-12)
        return self.wo.forward(np.einsum("bsht,bthd->bshd", self.p, ve).reshape(b, s, -1))

    def backward(self, dy):
        b, s, _ = dy.shape
        dy_o = self.wo.backward(dy).reshape(b, s, self.h, self.hd)
        ke, ve = np.repeat(self.kr, self.g, 2), np.repeat(self.v_raw, self.g, 2)
        dp = np.einsum("bshd,bthd->bsht", dy_o, ve)
        da = self.p * (dp - np.sum(self.p * dp, -1, keepdims=True)) * self.scale
        dqr = np.einsum("bsht,bthd->bshd", da, ke)
        dke_rep = np.einsum("bsht,bshd->bthd", da, self.qr)
        dve_rep = np.einsum("bsht,bshd->bthd", self.p, dy_o)
        dke = dke_rep.reshape(b, s, self.h//self.g, self.g, self.hd).sum(3)
        dve = dve_rep.reshape(b, s, self.h//self.g, self.g, self.hd).sum(3)
        dq = self.rope.apply(dqr, True).reshape(b, s, -1)
        dk = self.rope.apply(dke, True).reshape(b, s, -1)
        dv = dve.reshape(b, s, -1)
        return self.wq.backward(dq) + self.wk.backward(dk) + self.wv.backward(dv)

class SparseMoE:
    def __init__(self, d, n_experts=4, top_k=2):
        self.d, self.n_experts, self.top_k = d, n_experts, top_k
        self.gate = Linear(d, n_experts)
        self.experts = [[Linear(d, d*2), Linear(d*2, d)] for _ in range(n_experts)]

    def forward(self, x):
        self.x_shape = x.shape
        x_flat = x.reshape(-1, self.d)
        logits = self.gate.forward(x_flat)
        
        # Softmax gating
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Top-k selection
        top_k_indices = np.argsort(probs, axis=-1)[:, -self.top_k:]
        self.top_k_indices = top_k_indices
        
        # Masking and re-normalization
        mask = np.zeros_like(probs)
        np.put_along_axis(mask, top_k_indices, 1.0, axis=-1)
        self.gating_weights = (probs * mask)
        self.gating_weights /= (np.sum(self.gating_weights, axis=-1, keepdims=True) + 1e-12)
        
        out = np.zeros_like(x_flat)
        self.expert_inputs = []
        self.expert_intermediates = []
        
        for i in range(self.n_experts):
            idx = np.where(mask[:, i] == 1)[0]
            if len(idx) == 0:
                self.expert_inputs.append(None)
                self.expert_intermediates.append(None)
                continue
            
            ex_in = x_flat[idx]
            inter = silu(self.experts[i][0].forward(ex_in))
            ex_out = self.experts[i][1].forward(inter)
            
            out[idx] += ex_out * self.gating_weights[idx, i:i+1]
            self.expert_inputs.append(ex_in)
            self.expert_intermediates.append(inter)
            
        return out.reshape(self.x_shape)

    def backward(self, dy):
        dy_flat = dy.reshape(-1, self.d)
        x_flat = self.x_shape[0] * self.x_shape[1]
        dx_flat = np.zeros((x_flat, self.d), dtype="f4")
        dg_flat = np.zeros((x_flat, self.n_experts), dtype="f4")
        
        for i in range(self.n_experts):
            idx = np.where(self.gating_weights[:, i] > 0)[0]
            if len(idx) == 0: continue
            
            # Gradient w.r.t expert output
            de_out = dy_flat[idx] * self.gating_weights[idx, i:i+1]
            
            # Gradient w.r.t gating weight
            # expert_out_val = self.experts[i][1].forward(self.expert_intermediates[i])
            # Simplified: we need the expert output to compute d_gate. 
            # Re-calculating for memory efficiency in backward.
            ex_out_val = self.experts[i][1].forward(self.expert_intermediates[i])
            dg_flat[idx, i] = np.sum(dy_flat[idx] * ex_out_val, axis=-1)
            
            # Backprop through expert layers
            ds = self.experts[i][1].backward(de_out)
            dx_expert = self.experts[i][0].backward(ds * dsilu(self.experts[i][0].x))
            dx_flat[idx] += dx_expert
            
        # Backprop through gate softmax
        # Simplified d_softmax * d_topk_mask
        dg = self.gating_weights * (dg_flat - np.sum(self.gating_weights * dg_flat, axis=-1, keepdims=True))
        dx_flat += self.gate.backward(dg)
        
        return dx_flat.reshape(self.x_shape)

class RedundantConsensus:
    """Gemini-Groq Redundancy Logic: Parallel path validation"""
    def __init__(self, d):
        self.fast_path = Linear(d, d)
        self.norm = RMSNorm(d)
        
    def forward(self, x):
        self.identity = x
        return x + silu(self.fast_path.forward(self.norm.forward(x)))
    
    def backward(self, dy):
        dn = self.norm.backward(self.fast_path.backward(dy * dsilu(self.fast_path.x)))
        return dy + dn

class SovereignBlock:
    def __init__(self, d):
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn = GQA(d)
        self.moe = SparseMoE(d)
        self.consensus = RedundantConsensus(d)

    def forward(self, x):
        x = x + self.attn.forward(self.n1.forward(x))
        x = x + self.moe.forward(self.n2.forward(x))
        return self.consensus.forward(x)

    def backward(self, dy):
        dy = self.consensus.backward(dy)
        dy = dy + self.n2.backward(self.moe.backward(dy))
        return dy + self.n1.backward(self.attn.backward(dy))

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
        self.last_x = self.norm.forward(x[:, -1, :])
        return self.head.forward(self.last_x)

    def backward(self, dy):
        dy = self.norm.backward(self.head.backward(dy))
        dy_s = np.zeros((dy.shape[0], self.emb.x.shape[1], dy.shape[1]), "f4")
        dy_s[:, -1, :] = dy
        for b in reversed(self.blocks): dy_s = b.backward(dy_s)
        self.emb.backward(dy_s)

class AdamW:
    def __init__(self, model, lr=1e-3, wd=0.01, beta1=0.9, beta2=0.999):
        self.lr, self.wd, self.t = lr, wd, 0
        self.b1, self.b2 = beta1, beta2
        self.p = self._collect(model)
        self.m = [np.zeros_like(x) for x in self.p]
        self.v = [np.zeros_like(x) for x in self.p]

    def _collect(self, obj):
        ps = []
        if isinstance(obj, Linear): ps += [obj.W, obj.b]
        elif isinstance(obj, RMSNorm): ps += [obj.g]
        elif isinstance(obj, list): [ps.extend(self._collect(i)) for i in obj]
        elif hasattr(obj, "__dict__"):
            for k, v in obj.__dict__.items():
                if k not in ["rope", "x", "xn", "rms", "p", "qr", "kr", "v_raw", "top_k_indices", "gating_weights", "expert_inputs", "expert_intermediates"]:
                    ps.extend(self._collect(v))
        return ps

    def _grads(self, obj):
        gs = []
        if isinstance(obj, Linear): gs += [obj.dW, obj.db]
        elif isinstance(obj, RMSNorm): gs += [obj.dg]
        elif isinstance(obj, list): [gs.extend(self._grads(i)) for i in obj]
        elif hasattr(obj, "__dict__"):
            for k, v in obj.__dict__.items():
                if k not in ["rope", "x", "xn", "rms", "p", "qr", "kr", "v_raw", "top_k_indices", "gating_weights", "expert_inputs", "expert_intermediates"]:
                    gs.extend(self._grads(v))
        return gs

    def step(self, model):
        self.t += 1
        gs = self._grads(model)
        lr_t = self.lr * (np.sqrt(1 - self.b2**self.t) / (1 - self.b1**self.t))
        for i, (p, g) in enumerate(zip(self.p, gs)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g**2)
            p -= lr_t * (self.m[i] / (np.sqrt(self.v[i]) + 1e-8) + self.wd * p)

def train():
    # Synthetic 784-feature data (MNIST-like)
    N, D, C, BS, E = 256, 784, 10, 32, 50
    X = np.random.randn(N, D).astype("f4")
    Y = np.random.randint(0, C, N)
    
    model = OMEGA_ASI(D, 128, C, depth=2)
    optimizer = AdamW(model, lr=2e-3, wd=0.02)
    
    print(f"Starting Recursive Self-Evolution | Architecture: OMEGA-ASI | Features: {D}")
    
    for e in range(E):
        idx = np.random.permutation(N)
        epoch_loss, epoch_acc = 0, 0
        
        for i in range(0, N, BS):
            batch_idx = idx[i:i+BS]
            xb, yb = X[batch_idx], Y[batch_idx]
            
            # Forward
            logits = model.forward(xb)
            
            # Cross-entropy loss
            probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs /= np.sum(probs, axis=-1, keepdims=True)
            
            batch_loss = -np.mean(np.log(probs[np.arange(len(yb)), yb] + 1e-12))
            epoch_loss += batch_loss * len(yb)
            epoch_acc += np.sum(np.argmax(probs, axis=-1) == yb)
            
            # Backward
            dy = probs.copy()
            dy[np.arange(len(yb)), yb] -= 1
            model.backward(dy / len(yb))
            
            # Update
            optimizer.step(model)
            
        if (e + 1) % 5 == 0 or e == 0:
            print(f"Epoch {e+1:03d} | Loss: {epoch_loss/N:.4f} | Acc: {epoch_acc/N:.4f}")

if __name__ == "__main__":
    train()
