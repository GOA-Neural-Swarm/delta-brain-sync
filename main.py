import numpy as np

class Tensor:
    def __init__(self, data, name=""):
        self.data = data.astype("f4")
        self.grad = np.zeros_like(data, dtype="f4")
        self.name = name

class Linear:
    def __init__(self, i, o, name=""):
        scale = np.sqrt(2.0 / (i + o))
        self.w = Tensor(np.random.normal(0, scale, (i, o)), f"{name}_w")
        self.b = Tensor(np.zeros(o), f"{name}_b")

    def forward(self, x):
        self.x = x
        return x @ self.w.data + self.b.data

    def backward(self, dy):
        self.w.grad += self.x.reshape(-1, self.x.shape[-1]).T @ dy.reshape(-1, dy.shape[-1])
        self.b.grad += dy.reshape(-1, dy.shape[-1]).sum(0)
        return dy @ self.w.data.T

class RMSNorm:
    def __init__(self, d, e=1e-6):
        self.g = Tensor(np.ones(d), "norm_g")
        self.e = e

    def forward(self, x):
        self.x = x
        self.var = np.mean(x**2, axis=-1, keepdims=True)
        self.inv_std = 1.0 / np.sqrt(self.var + self.e)
        self.norm_x = x * self.inv_std
        return self.g.data * self.norm_x

    def backward(self, dy):
        d_norm = dy * self.g.data
        self.g.grad += np.sum(dy * self.norm_x, axis=tuple(range(dy.ndim - 1)))
        n = self.x.shape[-1]
        dx = (d_norm - self.norm_x * np.mean(d_norm * self.norm_x, axis=-1, keepdims=True)) * self.inv_std
        return dx

class SwiGLU:
    def forward(self, x):
        x, gate = np.split(x, 2, axis=-1)
        self.sig = 1.0 / (1.0 + np.exp(-np.clip(gate, -15, 15)))
        self.swish = gate * self.sig
        self.x, self.gate = x, gate
        return x * self.swish

    def backward(self, dy):
        dx = dy * self.swish
        dgate = dy * self.x * self.sig * (1.0 + self.gate * (1.0 - self.sig))
        return np.concatenate([dx, dgate], axis=-1)

class GQA:
    def __init__(self, d, h=8, g=2):
        self.d, self.h, self.g, self.hd = d, h, g, d // h
        self.wq = Linear(d, d, "q")
        self.wk = Linear(d, (h // g) * self.hd, "k")
        self.wv = Linear(d, (h // g) * self.hd, "v")
        self.wo = Linear(d, d, "o")
        self.scale = self.hd**-0.5

    def forward(self, x):
        b, s, _ = x.shape
        q = self.wq.forward(x).reshape(b, s, self.h, self.hd)
        k = self.wk.forward(x).reshape(b, s, self.h // self.g, self.hd)
        v = self.wv.forward(x).reshape(b, s, self.h // self.g, self.hd)
        
        # RoPE-lite
        pos = np.arange(s)[:, None]
        freqs = 10000 ** -(np.arange(0, self.hd, 2) / self.hd)
        args = pos * freqs
        cos, sin = np.cos(args), np.sin(args)
        
        def apply_rope(t):
            t_re, t_im = t[..., ::2], t[..., 1::2]
            return np.stack([t_re * cos[:, None, :] - t_im * sin[:, None, :], 
                            t_re * sin[:, None, :] + t_im * cos[:, None, :]], axis=-1).reshape(t.shape)

        self.q_rope, self.k_rope = apply_rope(q), apply_rope(k)
        self.v_raw = v
        
        k_rep = np.repeat(self.k_rope, self.g, axis=2)
        v_rep = np.repeat(v, self.g, axis=2)
        
        attn = np.einsum("bshd,bthd->bsht", self.q_rope, k_rep) * self.scale
        attn_max = np.max(attn, axis=-1, keepdims=True)
        exp_attn = np.exp(attn - attn_max)
        self.probs = exp_attn / (np.sum(exp_attn, axis=-1, keepdims=True) + 1e-12)
        
        out = np.einsum("bsht,bthd->bshd", self.probs, v_rep).reshape(b, s, -1)
        return self.wo.forward(out)

    def backward(self, dy):
        b, s = dy.shape[:2]
        dy_wo = self.wo.backward(dy).reshape(b, s, self.h, self.hd)
        
        k_rep = np.repeat(self.k_rope, self.g, axis=2)
        v_rep = np.repeat(self.v_raw, self.g, axis=2)
        
        d_probs = np.einsum("bshd,bthd->bsht", dy_wo, v_rep)
        d_attn = self.probs * (d_probs - np.sum(self.probs * d_probs, axis=-1, keepdims=True)) * self.scale
        
        dq_rope = np.einsum("bsht,bthd->bshd", d_attn, k_rep)
        dk_rep = np.einsum("bsht,bshd->bthd", d_attn, self.q_rope)
        dv_rep = np.einsum("bsht,bshd->bthd", self.probs, dy_wo)
        
        dk_rope = dk_rep.reshape(b, s, self.h // self.g, self.g, self.hd).sum(3)
        dv = dv_rep.reshape(b, s, self.h // self.g, self.g, self.hd).sum(3)
        
        # RoPE backward (approximate for speed)
        dq = self.wq.backward(dq_rope.reshape(b, s, -1))
        dk = self.wk.backward(dk_rope.reshape(b, s, -1))
        dv = self.wv.backward(dv.reshape(b, s, -1))
        return dq + dk + dv

class MoE:
    def __init__(self, d, n=4, k=2):
        self.d, self.n, self.k = d, n, k
        self.gate = Linear(d, n, "gate")
        self.experts_w1 = [Tensor(np.random.normal(0, np.sqrt(2/(d+d*2)), (d, d*2))) for _ in range(n)]
        self.experts_w2 = [Tensor(np.random.normal(0, np.sqrt(2/(d+d)), (d, d))) for _ in range(n)]
        self.swiglus = [SwiGLU() for _ in range(n)]

    def forward(self, x):
        orig_shape = x.shape
        x = x.reshape(-1, self.d)
        logits = self.gate.forward(x)
        
        # Groq-style deterministic routing
        probs = np.exp(logits - np.max(logits, -1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        
        self.indices = np.argsort(probs, axis=-1)[:, -self.k:]
        self.weights = np.take_along_axis(probs, self.indices, axis=-1)
        self.weights /= self.weights.sum(-1, keepdims=True) + 1e-12 # Gemini-style normalization
        
        out = np.zeros_like(x)
        self.expert_inputs = [[] for _ in range(self.n)]
        self.expert_caches = [[] for _ in range(self.n)]
        
        for i in range(self.n):
            mask = np.any(self.indices == i, axis=-1)
            if not np.any(mask): continue
            
            idx_in_topk = np.where(self.indices[mask] == i)[1]
            w = self.weights[mask, idx_in_topk][:, None]
            
            xi = x[mask]
            h1 = xi @ self.experts_w1[i].data
            act = self.swiglus[i].forward(h1)
            h2 = act @ self.experts_w2[i].data
            
            out[mask] += h2 * w
            self.expert_inputs[i] = (mask, xi, act, w, idx_in_topk)
            
        return out.reshape(orig_shape)

    def backward(self, dy):
        dy_flat = dy.reshape(-1, self.d)
        dx = np.zeros_like(dy_flat)
        dg = np.zeros((dy_flat.shape[0], self.n))
        
        for i in range(self.n):
            if not len(self.expert_inputs[i]): continue
            mask, xi, act, w, idx_in_topk = self.expert_inputs[i]
            
            dyi = dy_flat[mask] * w
            dg[mask, i] = np.sum(dy_flat[mask] * (act @ self.experts_w2[i].data), axis=-1)
            
            self.experts_w2[i].grad += act.T @ dyi
            dact = dyi @ self.experts_w2[i].data.T
            dh1 = self.swiglus[i].backward(dact)
            self.experts_w1[i].grad += xi.T @ dh1
            dx[mask] += dh1 @ self.experts_w1[i].data.T
            
        d_gate = self.gate.backward(dg - np.mean(dg, -1, keepdims=True))
        return (dx + d_gate).reshape(dy.shape)

class SovereignBlock:
    def __init__(self, d):
        self.norm1 = RMSNorm(d)
        self.attn = GQA(d)
        self.norm2 = RMSNorm(d)
        self.moe = MoE(d)
        # Gemini-style redundant path
        self.path_gate = Tensor(np.ones(1) * 0.5, "path_gate")

    def forward(self, x):
        res = x
        x = self.norm1.forward(x)
        x = self.attn.forward(x)
        x = res + x
        
        res = x
        x = self.norm2.forward(x)
        # Redundant logic: blend MoE with identity to stabilize early training
        moe_out = self.moe.forward(x)
        x = res + (self.path_gate.data * moe_out + (1 - self.path_gate.data) * x)
        return x

    def backward(self, dy):
        # Simplified path_gate backward
        d_moe = dy * self.path_gate.data
        dx_moe = self.moe.backward(self.norm2.backward(d_moe))
        dy = dy + dx_moe
        dy = dy + self.attn.backward(self.norm1.backward(dy))
        return dy

class OMEGA_ASI:
    def __init__(self, d_in, d_model, d_out, depth=2):
        self.embed = Linear(d_in, d_model, "embed")
        self.blocks = [SovereignBlock(d_model) for _ in range(depth)]
        self.final_norm = RMSNorm(d_model)
        self.head = Linear(d_model, d_out, "head")
        self.params = self._collect_params()

    def _collect_params(self):
        ps = []
        def _walk(obj):
            if isinstance(obj, (Linear, RMSNorm, Tensor)):
                if isinstance(obj, Tensor): ps.append(obj)
                else: ps.extend([obj.w, obj.b] if hasattr(obj, 'w') else [obj.g])
            elif isinstance(obj, MoE):
                ps.extend([obj.gate.w, obj.gate.b])
                ps.extend(obj.experts_w1)
                ps.extend(obj.experts_w2)
            elif hasattr(obj, "__dict__"):
                for v in obj.__dict__.values():
                    if isinstance(v, list): [_walk(i) for i in v]
                    else: _walk(v)
        _walk(self)
        return ps

    def forward(self, x):
        if x.ndim == 2: x = x[:, None, :]
        x = self.embed.forward(x)
        for b in self.blocks: x = b.forward(x)
        x = self.final_norm.forward(x[:, -1, :])
        return self.head.forward(x)

    def backward(self, dy):
        dy = self.head.backward(dy)
        dy = self.final_norm.backward(dy)
        db = np.zeros((dy.shape[0], self.embed.x.shape[1], dy.shape[1]), "f4")
        db[:, -1, :] = dy
        for b in reversed(self.blocks): db = b.backward(db)
        self.embed.backward(db)

class AdamW:
    def __init__(self, params, lr=1e-3, wd=0.01, b1=0.9, b2=0.999):
        self.params = params
        self.lr, self.wd, self.b1, self.b2 = lr, wd, b1, b2
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        lr_t = self.lr * (np.sqrt(1 - self.b2**self.t) / (1 - self.b1**self.t))
        for i, p in enumerate(self.params):
            # Gradient clipping
            grad = np.clip(p.grad, -1.0, 1.0)
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * grad
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (grad**2)
            p.data -= lr_t * (self.m[i] / (np.sqrt(self.v[i]) + 1e-8) + self.wd * p.data)
            p.grad.fill(0)

def train():
    N, D, C, BS, E = 1024, 784, 10, 64, 50
    X = np.random.randn(N, D).astype("f4")
    Y = np.random.randint(0, C, N)

    model = OMEGA_ASI(D, 128, C, depth=2)
    optimizer = AdamW(model.params, lr=3e-3)

    for epoch in range(E):
        idx = np.random.permutation(N)
        losses, accs = [], []
        
        for i in range(0, N, BS):
            xb, yb = X[idx[i:i+BS]], Y[idx[i:i+BS]]
            
            logits = model.forward(xb)
            probs = np.exp(logits - np.max(logits, -1, keepdims=True))
            probs /= probs.sum(-1, keepdims=True)
            
            loss = -np.mean(np.log(probs[np.arange(len(yb)), yb] + 1e-12))
            losses.append(loss)
            accs.append(np.mean(np.argmax(probs, -1) == yb))
            
            d_logits = probs.copy()
            d_logits[np.arange(len(yb)), yb] -= 1
            model.backward(d_logits / len(yb))
            optimizer.step()
            
        if (epoch + 1) % 5 == 0:
            print(f"EPOCH {epoch+1:03d} | LOSS {np.mean(losses):.4f} | ACC {np.mean(accs):.4f}")

if __name__ == "__main__":
    train()
