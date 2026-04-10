import numpy as np
import time

def softmax(x, axis=-1):
    ex = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return ex / (np.sum(ex, axis=axis, keepdims=True) + 1e-10)

class Linear:
    def __init__(self, in_d, out_d, std=None):
        scale = std if std else np.sqrt(2.0 / in_d)
        self.W = np.random.normal(0, scale, (in_d, out_d)).astype(np.float32)
        self.b = np.zeros(out_d, dtype=np.float32)
        self.x, self.dW, self.db = None, None, None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0)
        return dout @ self.W.T

    def params(self):
        return [{"ref": self.W, "grad": self.dW}, {"ref": self.b, "grad": self.db}]

class RMSNorm:
    def __init__(self, dim, eps=1e-6):
        self.scale = np.ones(dim, dtype=np.float32)
        self.eps = eps
        self.x, self.rstd, self.dscale = None, None, None

    def forward(self, x):
        self.x = x
        var = np.mean(x**2, axis=-1, keepdims=True)
        self.rstd = 1.0 / np.sqrt(var + self.eps)
        return self.scale * (x * self.rstd)

    def backward(self, dout):
        x_rstd = self.x * self.rstd
        self.dscale = np.sum(dout * x_rstd, axis=0)
        dx_rstd = dout * self.scale
        m_dx_rstd_x_rstd = np.mean(dx_rstd * x_rstd, axis=-1, keepdims=True)
        return self.rstd * (dx_rstd - x_rstd * m_dx_rstd_x_rstd)

    def params(self):
        return [{"ref": self.scale, "grad": self.dscale}]

class SwiGLU:
    def __init__(self, dim, h_dim):
        self.w1 = Linear(dim, h_dim)
        self.w2 = Linear(dim, h_dim)
        self.w3 = Linear(h_dim, dim)
        self.z1, self.z2, self.sig, self.swish = None, None, None, None

    def forward(self, x):
        self.z1 = self.w1.forward(x)
        self.z2 = self.w2.forward(x)
        self.sig = 1.0 / (1.0 + np.exp(-np.clip(self.z1, -10, 10)))
        self.swish = self.z1 * self.sig
        return self.w3.forward(self.swish * self.z2)

    def backward(self, dout):
        dz3 = self.w3.backward(dout)
        dz2 = dz3 * self.swish
        dswish = dz3 * self.z2
        dz1 = dswish * (self.sig * (1.0 + self.z1 * (1.0 - self.sig)))
        return self.w1.backward(dz1) + self.w2.backward(dz2)

    def params(self):
        return self.w1.params() + self.w2.params() + self.w3.params()

class MultiHeadAttention:
    def __init__(self, dim, heads=8):
        self.dim, self.heads = dim, heads
        self.head_dim = dim // heads
        self.wq = Linear(dim, dim)
        self.wk = Linear(dim, dim)
        self.wv = Linear(dim, dim)
        self.wo = Linear(dim, dim)
        self.q, self.k, self.v, self.attn, self.ctx = None, None, None, None, None

    def forward(self, x):
        b, d = x.shape
        q = self.wq.forward(x).reshape(b, self.heads, self.head_dim)
        k = self.wk.forward(x).reshape(b, self.heads, self.head_dim)
        v = self.wv.forward(x).reshape(b, self.heads, self.head_dim)
        self.q, self.k, self.v = q, k, v
        
        # Simplified self-attention for non-sequential data
        scores = np.einsum('bhd,bhd->bh', q, k) / np.sqrt(self.head_dim)
        self.attn = softmax(scores, axis=-1)[:, :, np.newaxis]
        self.ctx = (self.attn * v).reshape(b, d)
        return self.wo.forward(self.ctx)

    def backward(self, dout):
        dctx = self.wo.backward(dout)
        b, d = dctx.shape
        dctx_reshaped = dctx.reshape(b, self.heads, self.head_dim)
        
        dattn = np.sum(dctx_reshaped * self.v, axis=-1, keepdims=True)
        dv = dctx_reshaped * self.attn
        
        dscores = self.attn * (dattn - np.sum(self.attn * dattn, axis=1, keepdims=True))
        dscores /= np.sqrt(self.head_dim)
        
        dq = (dscores * self.k.mean(axis=-1, keepdims=True)).reshape(b, d) # Approximation for flat input
        dk = (dscores * self.q.mean(axis=-1, keepdims=True)).reshape(b, d)
        
        return self.wq.backward(dq) + self.wk.backward(dk) + self.wv.backward(dv.reshape(b, d))

    def params(self):
        return self.wq.params() + self.wk.params() + self.wv.params() + self.wo.params()

class SovereignMoE:
    def __init__(self, dim):
        self.gemini = SwiGLU(dim, dim * 4)
        self.groq = SwiGLU(dim, dim * 4)
        self.gate = Linear(dim, 2)
        self.probs, self.o1, self.o2 = None, None, None

    def forward(self, x):
        g = self.gate.forward(x)
        self.probs = softmax(g)
        self.o1 = self.gemini.forward(x)
        self.o2 = self.groq.forward(x)
        return self.probs[:, 0:1] * self.o1 + self.probs[:, 1:2] * self.o2

    def backward(self, dout):
        p0, p1 = self.probs[:, 0:1], self.probs[:, 1:2]
        dg_in = self.gemini.backward(dout * p0)
        dr_in = self.groq.backward(dout * p1)
        
        dp0 = np.sum(dout * self.o1, axis=-1, keepdims=True)
        dp1 = np.sum(dout * self.o2, axis=-1, keepdims=True)
        dg_raw = np.concatenate([dp0, dp1], axis=-1)
        d_gate = self.probs * (dg_raw - np.sum(self.probs * dg_raw, axis=-1, keepdims=True))
        
        return dg_in + dr_in + self.gate.backward(d_gate)

    def params(self):
        return self.gemini.params() + self.groq.params() + self.gate.params()

class SovereignBlock:
    def __init__(self, dim):
        self.ln1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim)
        self.ln2 = RMSNorm(dim)
        self.moe = SovereignMoE(dim)
        self.g1 = np.full(dim, 0.1, dtype=np.float32)
        self.g2 = np.full(dim, 0.1, dtype=np.float32)
        self.dg1, self.dg2 = None, None
        self.res1, self.res2 = None, None

    def forward(self, x):
        self.res1 = self.attn.forward(self.ln1.forward(x))
        x = x + self.g1 * self.res1
        self.res2 = self.moe.forward(self.ln2.forward(x))
        return x + self.g2 * self.res2

    def backward(self, dout):
        self.dg2 = np.sum(dout * self.res2, axis=0)
        dmoe = self.moe.backward(dout * self.g2)
        dln2 = self.ln2.backward(dmoe)
        dx_mid = dout + dln2
        self.dg1 = np.sum(dx_mid * self.res1, axis=0)
        dattn = self.attn.backward(dx_mid * self.g1)
        dln1 = self.ln1.backward(dattn)
        return dx_mid + dln1

    def params(self):
        p = self.ln1.params() + self.attn.params() + self.ln2.params() + self.moe.params()
        p.extend([{"ref": self.g1, "grad": self.dg1}, {"ref": self.g2, "grad": self.dg2}])
        return p

class SovereignArchitect:
    def __init__(self, in_d, h_d, out_d, depth=3):
        self.stem = Linear(in_d, h_d)
        self.blocks = [SovereignBlock(h_d) for _ in range(depth)]
        self.norm = RMSNorm(h_d)
        self.head = Linear(h_d, out_d)

    def forward(self, x):
        x = self.stem.forward(x)
        for b in self.blocks: x = b.forward(x)
        return self.head.forward(self.norm.forward(x))

    def backward(self, dout):
        dout = self.norm.backward(self.head.backward(dout))
        for b in reversed(self.blocks): dout = b.backward(dout)
        self.stem.backward(dout)

    def params(self):
        p = self.stem.params()
        for b in self.blocks: p.extend(b.params())
        p.extend(self.norm.params() + self.head.params())
        return p

class Lion:
    def __init__(self, params, lr=1e-4, b1=0.9, b2=0.99, wd=0.01):
        self.params = params
        self.lr, self.b1, self.b2, self.wd = lr, b1, b2, wd
        self.m = [np.zeros_like(p["ref"]) for p in params]

    def step(self, scale=1.0):
        lr = self.lr * scale
        for i, p in enumerate(self.params):
            if p["grad"] is None: continue
            param, grad = p["ref"], p["grad"]
            if self.wd > 0: param -= lr * self.wd * param
            update = np.sign(self.b1 * self.m[i] + (1.0 - self.b1) * grad)
            param -= lr * update
            self.m[i] = self.b2 * self.m[i] + (1.0 - self.b2) * grad

def evolve():
    N, D, K = 10000, 784, 10
    X = np.random.randn(N, D).astype(np.float32)
    y = np.random.randint(0, K, N)
    centers = np.random.randn(K, D).astype(np.float32) * 4.0
    X += centers[y]

    model = SovereignArchitect(D, 128, K, depth=3)
    opt = Lion(model.params(), lr=1e-4, wd=0.01)
    
    bs, epochs = 128, 40
    print("OMEGA-ASI | RECURSIVE SELF-EVOLUTION | V10-ULTRA")

    for ep in range(epochs):
        idx = np.random.permutation(N)
        l_sum, a_sum = 0, 0
        t0 = time.time()
        sched = 0.5 * (1 + np.cos(np.pi * ep / epochs))
        if ep < 5: sched *= (ep + 1) / 5

        for i in range(0, N, bs):
            bi = idx[i : i + bs]
            xb, yb = X[bi], y[bi]
            m = xb.shape[0]

            logits = model.forward(xb)
            probs = softmax(logits)
            
            loss = -np.mean(np.log(probs[range(m), yb] + 1e-10))
            l_sum += loss * (m / N)
            a_sum += np.mean(np.argmax(probs, axis=1) == yb) * (m / N)

            dout = probs.copy()
            dout[range(m), yb] -= 1
            model.backward(dout / m)

            gn = np.sqrt(sum(np.sum(p["grad"]**2) for p in model.params() if p["grad"] is not None))
            if gn > 1.0:
                for p in model.params():
                    if p["grad"] is not None: p["grad"] /= (gn + 1e-6)

            opt.step(scale=sched)

        dt = time.time() - t0
        print(f"EP:{ep:02d} | LOSS:{l_sum:.4f} | ACC:{a_sum:.4f} | {N/dt:.0f}s/s | LR:{opt.lr*sched:.6f}")

if __name__ == "__main__":
    evolve()
