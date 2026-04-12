import numpy as np
import time


def fast_softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exps = np.exp(x - x_max)
    return exps / (np.sum(exps, axis=axis, keepdims=True) + 1e-10)


def swiglu(x):
    h = x.shape[-1] // 2
    gate, val = x[..., :h], x[..., h:]
    return (gate / (1.0 + np.exp(-gate))) * val


def d_swiglu(x, dout):
    h = x.shape[-1] // 2
    gate, val = x[..., :h], x[..., h:]
    sig = 1.0 / (1.0 + np.exp(-gate))
    swish = gate * sig
    d_gate = (sig * (1.0 + gate * (1.0 - sig))) * val * dout
    d_val = swish * dout
    return np.concatenate([d_gate, d_val], axis=-1)


class Linear:
    def __init__(self, in_d, out_d, name=""):
        scale = np.sqrt(2.0 / (in_d + out_d))
        self.W = np.random.randn(in_d, out_d).astype(np.float32) * scale
        self.b = np.zeros(out_d, dtype=np.float32)
        self.dW, self.db = None, None
        self.x = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        x_2d = self.x.reshape(-1, self.x.shape[-1])
        dout_2d = dout.reshape(-1, dout.shape[-1])
        self.dW = np.dot(x_2d.T, dout_2d)
        self.db = np.sum(dout_2d, axis=0)
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
    def __init__(self, dim, max_len=1024):
        inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2).astype(np.float32) / dim))
        t = np.arange(max_len, dtype=np.float32)
        freqs = np.outer(t, inv_freq)
        emb = np.concatenate((freqs, freqs), axis=-1)
        self.cos = np.cos(emb)
        self.sin = np.sin(emb)

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
        dx_rot = dout * s_
        dx = dout * c
        dx[..., :half] += dx_rot[..., half:]
        dx[..., half:] -= dx_rot[..., :half]
        return dx


class SovereignGQA:
    def __init__(self, dim, heads=8, kv_heads=2):
        self.dim, self.heads, self.kv_heads = dim, heads, kv_heads
        self.head_dim = dim // heads
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

        # Grouped Query Attention logic
        k_rep = np.repeat(self.k_rope, self.heads // self.kv_heads, axis=2)
        v_rep = np.repeat(v, self.heads // self.kv_heads, axis=2)

        attn_logits = np.einsum("bshd,bthd->bsht", self.q_rope, k_rep) * self.scale
        self.probs = fast_softmax(attn_logits)
        out = np.einsum("bsht,bthd->bshd", self.probs, v_rep)
        return self.o_proj.forward(out.reshape(b, s, d))

    def backward(self, dout):
        b, s, d = dout.shape
        dout_o = self.o_proj.backward(dout).reshape(b, s, self.heads, self.head_dim)

        k_rep = np.repeat(self.k_rope, self.heads // self.kv_heads, axis=2)
        v_rep = np.repeat(self.v_val, self.heads // self.kv_heads, axis=2)

        d_probs = np.einsum("bshd,bthd->bsht", dout_o, v_rep)
        d_v_rep = np.einsum("bsht,bshd->bthd", self.probs, dout_o)

        d_logits = (
            self.probs
            * (d_probs - np.sum(self.probs * d_probs, axis=-1, keepdims=True))
            * self.scale
        )

        dq_rope = np.einsum("bsht,bthd->bshd", d_logits, k_rep)
        dk_rep = np.einsum("bsht,bshd->bthd", d_logits, self.q_rope)

        dq = self.rope.backward(dq_rope)
        dk_rope = np.sum(dk_rep.reshape(b, s, self.kv_heads, -1, self.head_dim), axis=3)
        dk = self.rope.backward(dk_rope)
        dv = np.sum(d_v_rep.reshape(b, s, self.kv_heads, -1, self.head_dim), axis=3)

        return (
            self.q_proj.backward(dq.reshape(b, s, d))
            + self.k_proj.backward(dk.reshape(b, s, -1))
            + self.v_proj.backward(dv.reshape(b, s, -1))
        )


class GroqMoE:
    def __init__(self, dim, num_experts=4, top_k=2):
        self.dim, self.num_experts, self.top_k = dim, num_experts, top_k
        self.gate = Linear(dim, num_experts)
        self.experts = [
            [Linear(dim, dim * 2), Linear(dim, dim)] for _ in range(num_experts)
        ]

    def forward(self, x):
        self.orig_shape = x.shape
        x_flat = x.reshape(-1, self.dim)
        logits = self.gate.forward(x_flat)
        probs = fast_softmax(logits)

        indices = np.argsort(probs, axis=-1)[:, -self.top_k :]
        self.indices, self.probs = indices, probs

        out_flat = np.zeros_like(x_flat)
        self.expert_ctx = []

        for i in range(self.num_experts):
            mask = np.any(indices == i, axis=1)
            if not np.any(mask):
                self.expert_ctx.append(None)
                continue

            e_in = x_flat[mask]
            h = self.experts[i][0].forward(e_in)
            act = swiglu(h)
            e_out = self.experts[i][1].forward(act)

            p = probs[mask, i : i + 1]
            out_flat[mask] += e_out * p
            self.expert_ctx.append((mask, h, act, e_out))

        return out_flat.reshape(self.orig_shape)

    def backward(self, dout):
        dout_flat = dout.reshape(-1, self.dim)
        dx_flat = np.zeros((dout_flat.shape[0], self.dim), dtype=np.float32)
        d_logits = np.zeros_like(self.probs)

        for i in range(self.num_experts):
            if self.expert_ctx[i] is None:
                continue
            mask, h, act, e_out = self.expert_ctx[i]

            p = self.probs[mask, i : i + 1]
            de_out = dout_flat[mask] * p
            d_logits[mask, i] = np.sum(dout_flat[mask] * e_out, axis=-1)

            d_act = self.experts[i][1].backward(de_out)
            dh = d_swiglu(h, d_act)
            dx_flat[mask] += self.experts[i][0].backward(dh)

        dg = self.probs * (
            d_logits - np.sum(self.probs * d_logits, axis=-1, keepdims=True)
        )
        dx_flat += self.gate.backward(dg)
        return dx_flat.reshape(self.orig_shape)


class SovereignBlock:
    def __init__(self, dim):
        self.norm1 = RMSNorm(dim)
        self.attn = SovereignGQA(dim)
        self.norm2 = RMSNorm(dim)
        self.moe = GroqMoE(dim)

    def forward(self, x):
        self.res1 = x
        x = x + self.attn.forward(self.norm1.forward(x))
        self.res2 = x
        return x + self.moe.forward(self.norm2.forward(x))

    def backward(self, dout):
        dm_in = self.moe.backward(dout)
        dx = dout + self.norm2.backward(dm_in)
        da_in = self.attn.backward(dx)
        return dx + self.norm1.backward(da_in)


class OMEGA_ASI_V5:
    def __init__(self, in_dim=784, h_dim=128, out_dim=10, depth=4):
        self.patch_size = 16
        self.num_patches = in_dim // self.patch_size
        self.stem = Linear(self.patch_size, h_dim)
        self.pos_emb = (
            np.random.randn(1, self.num_patches, h_dim).astype(np.float32) * 0.02
        )
        self.blocks = [SovereignBlock(h_dim) for _ in range(depth)]
        self.norm = RMSNorm(h_dim)
        self.head = Linear(h_dim, out_dim)

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

        def walk(obj):
            if isinstance(obj, (Linear, RMSNorm)):
                params.append(obj)
            elif isinstance(obj, list):
                [walk(i) for i in obj]
            elif hasattr(obj, "__dict__"):
                [walk(v) for v in obj.__dict__.values()]

        walk(self)
        return params


class Lion:
    def __init__(self, params, lr=1e-4, b1=0.9, b2=0.99, wd=0.01):
        self.params, self.lr, self.b1, self.b2, self.wd = params, lr, b1, b2, wd
        self.m = [
            np.zeros_like(p.W) if hasattr(p, "W") else np.zeros_like(p.scale)
            for p in params
        ]
        self.m_b = [np.zeros_like(p.b) if hasattr(p, "b") else None for p in params]

    def step(self):
        for i, p in enumerate(self.params):
            if hasattr(p, "W"):
                for attr, m_list in [("W", self.m), ("b", self.m_b)]:
                    g = getattr(p, "d" + attr)
                    m = m_list[i]
                    u = np.sign(self.b1 * m + (1.0 - self.b1) * g)
                    w = getattr(p, attr)
                    w -= self.lr * (u + self.wd * w if attr == "W" else u)
                    m_list[i] = self.b2 * m + (1.0 - self.b2) * g
                    setattr(p, attr, w)
            else:
                u = np.sign(self.b1 * self.m[i] + (1.0 - self.b1) * p.d_scale)
                p.scale -= self.lr * (u + self.wd * p.scale)
                self.m[i] = self.b2 * self.m[i] + (1.0 - self.b2) * p.d_scale


def get_data(n=2048):
    X = np.random.randn(n, 784).astype(np.float32)
    y = np.random.randint(0, 10, n)
    X[range(n), y * 78] += 10.0  # Synthetic correlation
    return (X - np.mean(X)) / (np.std(X) + 1e-6), y


def train():
    X, y = get_data(4096)
    model = OMEGA_ASI_V5(h_dim=64, depth=2)
    params = model.get_params()
    opt = Lion(params, lr=2e-4)
    bs = 64
    print("OMEGA-ASI V5 | ARCHITECT: SOVEREIGN | STATUS: ONLINE")
    for ep in range(15):
        idx = np.random.permutation(len(X))
        l_acc, a_acc, t0 = 0, 0, time.time()
        for i in range(0, len(X), bs):
            xb, yb = X[idx[i : i + bs]], y[idx[i : i + bs]]
            if len(xb) < bs:
                continue

            logits = model.forward(xb)
            probs = fast_softmax(logits)

            loss = -np.mean(np.log(probs[range(bs), yb] + 1e-10))
            l_acc += loss * bs
            a_acc += np.sum(np.argmax(probs, axis=1) == yb)

            dout = probs.copy()
            dout[range(bs), yb] -= 1
            model.backward(dout / bs)

            # Global Gradient Clipping
            gn = np.sqrt(
                sum(
                    np.sum(p.dW**2) + np.sum(p.db**2)
                    for p in params
                    if hasattr(p, "dW")
                )
            )
            if gn > 1.0:
                for p in params:
                    if hasattr(p, "dW"):
                        p.dW /= gn
                        p.db /= gn
                    if hasattr(p, "d_scale"):
                        p.d_scale /= gn

            opt.step()

        dt = time.time() - t0
        print(
            f"EPOCH:{ep:02d} | LOSS:{l_acc/len(X):.4f} | ACC:{a_acc/len(X):.4f} | SPEED:{len(X)/dt:.1f} samples/s"
        )


if __name__ == "__main__":
    train()
