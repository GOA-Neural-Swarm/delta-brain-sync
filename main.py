import numpy as np
import time


def fast_softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


def silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -10, 10)))


def d_silu(x):
    s = 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))
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
        self.dW = np.dot(
            self.x.reshape(-1, self.x.shape[-1]).T, dout.reshape(-1, dout.shape[-1])
        )
        self.db = np.sum(dout, axis=tuple(range(len(dout.shape) - 1)))
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


class ExpertGemini:
    def __init__(self, dim):
        self.w1 = Linear(dim, dim * 2)
        self.w2 = Linear(dim, dim * 2)
        self.w3 = Linear(dim * 2, dim)

    def forward(self, x):
        self.x1, self.x2 = self.w1.forward(x), self.w2.forward(x)
        self.silu_x1 = silu(self.x1)
        self.pre_out = self.silu_x1 * self.x2
        return self.w3.forward(self.pre_out)

    def backward(self, dout):
        d_w3 = self.w3.backward(dout)
        dx1 = d_w3 * self.x2 * d_silu(self.x1)
        dx2 = d_w3 * self.silu_x1
        return self.w1.backward(dx1) + self.w2.backward(dx2)


class ExpertGroq:
    def __init__(self, dim):
        self.w1 = Linear(dim, dim // 2)
        self.w2 = Linear(dim // 2, dim)

    def forward(self, x):
        self.z = self.w1.forward(x)
        return self.w2.forward(self.z)

    def backward(self, dout):
        return self.w1.backward(self.w2.backward(dout))


class BimodalMoE:
    def __init__(self, dim):
        self.gemini = ExpertGemini(dim)
        self.groq = ExpertGroq(dim)
        self.gate = Linear(dim, 2)

    def forward(self, x):
        self.x = x
        self.logits = self.gate.forward(x)
        self.probs = fast_softmax(self.logits)
        self.out_gem = self.gemini.forward(x)
        self.out_groq = self.groq.forward(x)
        return (
            self.probs[..., 0:1] * self.out_gem + self.probs[..., 1:2] * self.out_groq
        )

    def backward(self, dout):
        p1, p2 = self.probs[..., 0:1], self.probs[..., 1:2]
        dx_gem = self.gemini.backward(dout * p1)
        dx_groq = self.groq.backward(dout * p2)
        dp1 = np.sum(dout * self.out_gem, axis=-1, keepdims=True)
        dp2 = np.sum(dout * self.out_groq, axis=-1, keepdims=True)
        dp = np.concatenate([dp1, dp2], axis=-1)
        dg = self.probs * (dp - np.sum(self.probs * dp, axis=-1, keepdims=True))
        dx_gate = self.gate.backward(dg)
        return dx_gem + dx_groq + dx_gate


class MultiHeadAttention:
    def __init__(self, dim, heads=8):
        self.dim, self.heads, self.hd = dim, heads, dim // heads
        self.scale = 1.0 / np.sqrt(self.hd)
        self.wq, self.wk, self.wv, self.wo = [Linear(dim, dim) for _ in range(4)]

    def forward(self, x):
        b, s, d = x.shape
        self.q = (
            self.wq.forward(x).reshape(b, s, self.heads, self.hd).transpose(0, 2, 1, 3)
        )
        self.k = (
            self.wk.forward(x).reshape(b, s, self.heads, self.hd).transpose(0, 2, 1, 3)
        )
        self.v = (
            self.wv.forward(x).reshape(b, s, self.heads, self.hd).transpose(0, 2, 1, 3)
        )
        self.dots = np.matmul(self.q, self.k.transpose(0, 1, 3, 2)) * self.scale
        self.att = fast_softmax(self.dots)
        out = np.matmul(self.att, self.v).transpose(0, 2, 1, 3).reshape(b, s, d)
        return self.wo.forward(out)

    def backward(self, dout):
        b, s, d = dout.shape
        d_wo = (
            self.wo.backward(dout)
            .reshape(b, s, self.heads, self.hd)
            .transpose(0, 2, 1, 3)
        )
        d_att = np.matmul(d_wo, self.v.transpose(0, 1, 3, 2))
        d_v = np.matmul(self.att.transpose(0, 1, 3, 2), d_wo)
        d_dots = (
            self.att
            * (d_att - np.sum(self.att * d_att, axis=-1, keepdims=True))
            * self.scale
        )
        d_q = np.matmul(d_dots, self.k)
        d_k = np.matmul(d_dots.transpose(0, 1, 3, 2), self.q)
        dq = self.wq.backward(d_q.transpose(0, 2, 1, 3).reshape(b, s, d))
        dk = self.wk.backward(d_k.transpose(0, 2, 1, 3).reshape(b, s, d))
        dv = self.wv.backward(d_v.transpose(0, 2, 1, 3).reshape(b, s, d))
        return dq + dk + dv


class SovereignBlock:
    def __init__(self, dim):
        self.ln1, self.attn = RMSNorm(dim), MultiHeadAttention(dim)
        self.ln2, self.moe = RMSNorm(dim), BimodalMoE(dim)
        self.ls1 = np.ones(dim, dtype=np.float32) * 0.1
        self.ls2 = np.ones(dim, dtype=np.float32) * 0.1

    def forward(self, x):
        self.x = x
        self.a = self.attn.forward(self.ln1.forward(x))
        self.x2 = x + self.ls1 * self.a
        self.m = self.moe.forward(self.ln2.forward(self.x2))
        return self.x2 + self.ls2 * self.m

    def backward(self, dout):
        self.dls2 = np.sum(dout * self.m, axis=(0, 1))
        dm = self.moe.backward(dout * self.ls2)
        dn2 = self.ln2.backward(dm)
        dx2 = dout + dn2
        self.dls1 = np.sum(dx2 * self.a, axis=(0, 1))
        da = self.attn.backward(dx2 * self.ls1)
        dn1 = self.ln1.backward(da)
        return dx2 + dn1


class SovereignArchitect:
    def __init__(self, h_d, out_d, depth=4):
        self.patch_dim, self.num_patches = 16, 49
        self.stem = Linear(self.patch_dim, h_d)
        self.blocks = [SovereignBlock(h_d) for _ in range(depth)]
        self.norm = RMSNorm(h_d)
        self.head = Linear(h_d, out_d, init_scale=0.1)

    def forward(self, x):
        b = x.shape[0]
        x = x.reshape(b, self.num_patches, self.patch_dim)
        x = self.stem.forward(x)
        for block in self.blocks:
            x = block.forward(x)
        return self.head.forward(self.norm.forward(np.mean(x, axis=1)))

    def backward(self, dout):
        dout = self.norm.backward(self.head.backward(dout))
        dout = np.tile(
            dout[:, np.newaxis, :] / self.num_patches, (1, self.num_patches, 1)
        )
        for block in reversed(self.blocks):
            dout = block.backward(dout)
        self.stem.backward(dout)

    def get_params(self):
        params = []

        def walk(o):
            if isinstance(o, (Linear, RMSNorm)):
                params.append(o)
            elif hasattr(o, "ls1"):
                params.append(o)
            elif hasattr(o, "__dict__"):
                for v in o.__dict__.values():
                    if isinstance(v, list):
                        [walk(i) for i in v]
                    else:
                        walk(v)

        walk(self)
        return params


class LionOptimizer:
    def __init__(self, params, lr=1e-4, b1=0.9, b2=0.99, wd=0.01):
        self.params, self.lr, self.b1, self.b2, self.wd = params, lr, b1, b2, wd
        self.m = [
            np.zeros_like(getattr(p, "W", getattr(p, "g", getattr(p, "ls1", None))))
            for p in params
        ]

    def step(self, scale=1.0):
        lr = self.lr * scale
        for i, p in enumerate(self.params):
            if hasattr(p, "W"):
                for attr in ["W", "b"]:
                    w, g = getattr(p, attr), getattr(p, "d" + attr)
                    if attr == "W":
                        u = np.sign(self.b1 * self.m[i] + (1.0 - self.b1) * g)
                        w -= lr * (u + self.wd * w)
                        self.m[i] = self.b2 * self.m[i] + (1.0 - self.b2) * g
                    else:
                        w -= lr * np.sign(g)
                    setattr(p, attr, w)
            elif hasattr(p, "g"):
                u = np.sign(self.b1 * self.m[i] + (1.0 - self.b1) * p.dg)
                p.g -= lr * u
                self.m[i] = self.b2 * self.m[i] + (1.0 - self.b2) * p.dg
            elif hasattr(p, "ls1"):
                for a in ["ls1", "ls2"]:
                    setattr(p, a, getattr(p, a) - lr * np.sign(getattr(p, "d" + a)))


def generate_data(n=5000):
    X = np.random.randn(n, 784).astype(np.float32)
    y = np.random.randint(0, 10, n)
    centers = np.random.randn(10, 784).astype(np.float32) * 4.0
    X += centers[y]
    return (X - np.mean(X)) / (np.std(X) + 1e-6), y


def train():
    X, y = generate_data()
    model = SovereignArchitect(h_d=128, out_d=10, depth=3)
    p_list = model.get_params()
    opt = LionOptimizer(p_list, lr=3e-4, wd=0.05)
    bs, epochs = 64, 40

    print(f"OMEGA-ASI | V8-SOVEREIGN-CORE | ACTIVE")

    for ep in range(epochs):
        idx = np.random.permutation(len(X))
        l_acc, a_acc, t0 = 0, 0, time.time()
        sched = 0.5 * (1 + np.cos(np.pi * ep / epochs))

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

            gn = np.sqrt(
                sum(np.sum(getattr(p, "dW", getattr(p, "dg", 0))) ** 2 for p in p_list)
            )
            if gn > 1.0:
                for p in p_list:
                    if hasattr(p, "dW"):
                        p.dW /= gn
                        p.db /= gn
                    if hasattr(p, "dg"):
                        p.dg /= gn

            opt.step(scale=sched)

        dt = time.time() - t0
        print(
            f"EP:{ep:03d} | LOSS:{l_acc:.4f} | ACC:{a_acc:.4f} | SPEED:{len(X)/dt:.0f} samples/s"
        )


if __name__ == "__main__":
    train()
