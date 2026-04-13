import numpy as np
import time


def swiglu(x):
    h = x.shape[-1] // 2
    a, b = x[..., :h], x[..., h:]
    return a * (1.0 / (1.0 + np.exp(-np.clip(a, -10, 10)))) * b


def d_swiglu(x, dout):
    h = x.shape[-1] // 2
    a, b = x[..., :h], x[..., h:]
    sig = 1.0 / (1.0 + np.exp(-np.clip(a, -10, 10)))
    swi = a * sig
    da = dout * b * (sig + swi * (1.0 - sig))
    db = dout * swi
    return np.concatenate([da, db], axis=-1)


def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exps = np.exp(x - x_max)
    return exps / (np.sum(exps, axis=axis, keepdims=True) + 1e-10)


class Linear:
    def __init__(self, in_d, out_d, std=None):
        scale = std or np.sqrt(2.0 / in_d)
        self.W = np.random.randn(in_d, out_d).astype(np.float32) * scale
        self.b = np.zeros(out_d, dtype=np.float32)
        self.x, self.dW, self.db = None, None, None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        x_flat = self.x.reshape(-1, self.x.shape[-1])
        d_flat = dout.reshape(-1, dout.shape[-1])
        self.dW = np.dot(x_flat.T, d_flat)
        self.db = np.sum(d_flat, axis=0)
        return np.dot(dout, self.W.T)


class RMSNorm:
    def __init__(self, dim, eps=1e-6):
        self.g = np.ones(dim, dtype=np.float32)
        self.eps, self.x, self.inv_rms = eps, None, None

    def forward(self, x):
        self.x = x
        self.inv_rms = 1.0 / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return self.g * (x * self.inv_rms)

    def backward(self, dout):
        nx = self.x * self.inv_rms
        self.dg = np.sum(dout * nx, axis=tuple(range(len(dout.shape) - 1)))
        d_nx = dout * self.g
        return self.inv_rms * (d_nx - nx * np.mean(d_nx * nx, axis=-1, keepdims=True))


class RoPE:
    def __init__(self, dim, max_seq=2048):
        inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2).astype(np.float32) / dim))
        t = np.arange(max_seq)
        freqs = np.outer(t, inv_freq)
        emb = np.concatenate([freqs, freqs], axis=-1)
        self.cos = np.cos(emb)[None, :, None, :]
        self.sin = np.sin(emb)[None, :, None, :]

    def apply(self, x):
        s = x.shape[1]
        c, sn = self.cos[:, :s, :, :], self.sin[:, :s, :, :]
        x_rot = np.concatenate(
            [-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]], axis=-1
        )
        return x * c + x_rot * sn

    def backward(self, dout):
        s = dout.shape[1]
        c, sn = self.cos[:, :s, :, :], self.sin[:, :s, :, :]
        half = dout.shape[-1] // 2
        dx = dout * c
        dx[..., :half] += dout[..., half:] * sn[..., :half]
        dx[..., half:] -= dout[..., :half] * sn[..., half:]
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
        self.rope = RoPE(self.head_dim)
        self.scale = 1.0 / np.sqrt(self.head_dim)

    def forward(self, x):
        b, s, d = x.shape
        q = self.q_proj.forward(x).reshape(b, s, self.heads, self.head_dim)
        k = self.k_proj.forward(x).reshape(b, s, self.kv_heads, self.head_dim)
        v = self.v_proj.forward(x).reshape(b, s, self.kv_heads, self.head_dim)
        self.qr, self.kr, self.v_raw = self.rope.apply(q), self.rope.apply(k), v
        self.k_rep = np.repeat(self.kr, self.group, axis=2)
        self.v_rep = np.repeat(v, self.group, axis=2)
        attn = np.einsum("bshd,bthd->bsht", self.qr, self.k_rep) * self.scale
        self.probs = softmax(attn)
        out = np.einsum("bsht,bthd->bshd", self.probs, self.v_rep)
        return self.o_proj.forward(out.reshape(b, s, d))

    def backward(self, dout):
        b, s, d = dout.shape
        dout_o = self.o_proj.backward(dout).reshape(b, s, self.heads, self.head_dim)
        d_probs = np.einsum("bshd,bthd->bsht", dout_o, self.v_rep)
        d_v_rep = np.einsum("bsht,bshd->bthd", self.probs, dout_o)
        d_attn = (
            self.probs
            * (d_probs - np.sum(self.probs * d_probs, axis=-1, keepdims=True))
        ) * self.scale
        d_qr = np.einsum("bsht,bthd->bshd", d_attn, self.k_rep)
        d_kr_rep = np.einsum("bsht,bshd->bthd", d_attn, self.qr)
        dq = self.rope.backward(d_qr).reshape(b, s, d)
        dk_sum = np.sum(
            d_kr_rep.reshape(b, s, self.kv_heads, self.group, self.head_dim), axis=3
        )
        dk = self.rope.backward(dk_sum).reshape(b, s, -1)
        dv = np.sum(
            d_v_rep.reshape(b, s, self.kv_heads, self.group, self.head_dim), axis=3
        ).reshape(b, s, -1)
        return (
            self.q_proj.backward(dq)
            + self.k_proj.backward(dk)
            + self.v_proj.backward(dv)
        )


class GroqMoE:
    def __init__(self, dim, n_exp=8, top_k=2):
        self.dim, self.n_exp, self.top_k = dim, n_exp, top_k
        self.gate = Linear(dim, n_exp)
        self.experts = [
            [Linear(dim, dim * 2), Linear(dim * 2, dim)] for _ in range(n_exp)
        ]

    def forward(self, x):
        self.sh = x.shape
        xf = x.reshape(-1, self.dim)
        logits = self.gate.forward(xf)
        probs = softmax(logits)
        self.top_k_indices = np.argsort(probs, axis=-1)[:, -self.top_k :]
        self.top_k_probs = np.take_along_axis(probs, self.top_k_indices, axis=-1)
        self.top_k_probs /= np.sum(self.top_k_probs, axis=-1, keepdims=True) + 1e-10
        out = np.zeros_like(xf)
        self.expert_ctx = [[] for _ in range(self.n_exp)]
        for k in range(self.top_k):
            indices = self.top_k_indices[:, k]
            p_vals = self.top_k_probs[:, k : k + 1]
            for i in range(self.n_exp):
                mask = indices == i
                if not np.any(mask):
                    continue
                h = self.experts[i][0].forward(xf[mask])
                act = swiglu(h)
                eo = self.experts[i][1].forward(act)
                out[mask] += eo * p_vals[mask]
                self.expert_ctx[i].append((mask, h, act, eo, p_vals[mask]))
        self.probs = probs
        return out.reshape(self.sh)

    def backward(self, dout):
        df = dout.reshape(-1, self.dim)
        xf = self.gate.x
        dx = np.zeros_like(xf)
        dlg = np.zeros_like(self.probs)
        for i in range(self.n_exp):
            for mask, h, act, eo, p_val in self.expert_ctx[i]:
                deo = df[mask] * p_val
                dlg[mask, i] += np.sum(df[mask] * eo, axis=-1)
                dact = self.experts[i][1].backward(deo)
                dh = d_swiglu(h, dact)
                dx[mask] += self.experts[i][0].backward(dh)
        dg = self.probs * (dlg - np.sum(self.probs * dlg, axis=-1, keepdims=True))
        return (dx + self.gate.backward(dg)).reshape(self.sh)


class SovereignBlock:
    def __init__(self, dim):
        self.n1, self.attn = RMSNorm(dim), GeminiGQA(dim)
        self.n2, self.moe = RMSNorm(dim), GroqMoE(dim)

    def forward(self, x):
        self.x1 = x
        self.x2 = self.x1 + self.attn.forward(self.n1.forward(self.x1))
        self.x3 = self.x2 + self.moe.forward(self.n2.forward(self.x2))
        return self.x3

    def backward(self, dout):
        d_moe = self.moe.backward(dout)
        d_x2 = dout + self.n2.backward(d_moe)
        d_attn = self.attn.backward(d_x2)
        return d_x2 + self.n1.backward(d_attn)


class OMEGA_ASI_X10:
    def __init__(self, in_d=784, h_d=128, out_d=10, depth=2):
        self.stem = Linear(in_d, h_d)
        self.blocks = [SovereignBlock(h_d) for _ in range(depth)]
        self.norm = RMSNorm(h_d)
        self.head = Linear(h_d, out_d)

    def forward(self, x):
        x = self.stem.forward(x)[:, None, :]
        for b in self.blocks:
            x = b.forward(x)
        self.f = self.norm.forward(x[:, 0, :])
        return self.head.forward(self.f)

    def backward(self, dout):
        dout = self.norm.backward(self.head.backward(dout))[:, None, :]
        for b in reversed(self.blocks):
            dout = b.backward(dout)
        self.stem.backward(dout[:, 0, :])

    def get_params(self):
        p = []

        def coll(o):
            if isinstance(o, (Linear, RMSNorm)):
                p.append(o)
            elif hasattr(o, "__dict__"):
                for v in o.__dict__.values():
                    if isinstance(v, list):
                        [coll(i) for i in v]
                    else:
                        coll(v)

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
                    if m_list[i] is None:
                        continue
                    g = getattr(p, "d" + attr)
                    m = m_list[i]
                    u = np.sign(self.b1 * m + (1.0 - self.b1) * g)
                    w = getattr(p, attr)
                    w -= self.lr * (u + self.wd * w if attr == "W" else u)
                    m_list[i] = self.b2 * m + (1.0 - self.b2) * g
                    setattr(p, attr, w)
            else:
                m = self.m[i]
                u = np.sign(self.b1 * m + (1.0 - self.b1) * p.dg)
                p.g -= self.lr * (u + self.wd * p.g)
                self.m[i] = self.b2 * m + (1.0 - self.b2) * p.dg


def get_data(n=4096):
    X = np.random.randn(n, 784).astype(np.float32)
    y = np.random.randint(0, 10, n)
    for i in range(n):
        X[i, y[i] * 78 : (y[i] + 1) * 78] += 5.0
    return (X - np.mean(X)) / (np.std(X) + 1e-6), y


def train():
    X, y = get_data(4096)
    model = OMEGA_ASI_X10(h_d=128, depth=2)
    params = model.get_params()
    opt = Lion(params, lr=2e-4)
    bs, epochs = 64, 20
    for ep in range(epochs):
        idx = np.random.permutation(len(X))
        ls, acc, t0 = 0, 0, time.time()
        opt.lr = 2e-4 * 0.5 * (1 + np.cos(np.pi * ep / epochs))
        for i in range(0, len(X), bs):
            xb, yb = X[idx[i : i + bs]], y[idx[i : i + bs]]
            if len(xb) < bs:
                continue
            probs = softmax(model.forward(xb))
            ls += -np.mean(np.log(probs[range(bs), yb] + 1e-10)) * bs
            acc += np.sum(np.argmax(probs, axis=1) == yb)
            dout = probs.copy()
            dout[range(bs), yb] -= 1
            model.backward(dout / bs)
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
                    if hasattr(p, "dg"):
                        p.dg /= gn
            opt.step()
        print(
            f"EP:{ep:02d} | LOSS:{ls/len(X):.4f} | ACC:{acc/len(X):.4f} | {len(X)/(time.time()-t0):.1f} s/s"
        )


if __name__ == "__main__":
    train()
