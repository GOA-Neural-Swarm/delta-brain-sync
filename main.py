import numpy as np
import time

def stable_softmax(x, axis=-1):
    z = x - np.max(x, axis=axis, keepdims=True)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=axis, keepdims=True)
    return numerator / (denominator + 1e-12)

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
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return np.dot(dout, self.W.T)

    def params(self):
        return [{"ref": self.W, "grad": self.dW}, {"ref": self.b, "grad": self.db}]

class RMSNorm:
    def __init__(self, dim, eps=1e-6):
        self.scale = np.ones(dim, dtype=np.float32)
        self.eps = eps
        self.x, self.rstd, self.dscale = None, None, None

    def forward(self, x):
        self.x = x
        ms = np.mean(x**2, axis=-1, keepdims=True)
        self.rstd = 1.0 / np.sqrt(ms + self.eps)
        return self.scale * (x * self.rstd)

    def backward(self, dout):
        norm_x = self.x * self.rstd
        self.dscale = np.sum(dout * norm_x, axis=0)
        v = dout * self.scale
        return self.rstd * (v - norm_x * np.mean(v * norm_x, axis=-1, keepdims=True))

    def params(self):
        return [{"ref": self.scale, "grad": self.dscale}]

class SwiGLU:
    def __init__(self, dim, h_dim):
        self.w12 = Linear(dim, h_dim * 2)
        self.w3 = Linear(h_dim, dim)
        self.h_dim = h_dim
        self.x, self.gate, self.act = None, None, None

    def forward(self, x):
        self.x = x
        z = self.w12.forward(x)
        self.gate, self.act = z[:, :self.h_dim], z[:, self.h_dim:]
        # Sigmoid for Swish
        sig = 1.0 / (1.0 + np.exp(-np.clip(self.gate, -12, 12)))
        swish = self.gate * sig
        self.swish_sig = sig
        self.combined = swish * self.act
        return self.w3.forward(self.combined)

    def backward(self, dout):
        d_combined = self.w3.backward(dout)
        d_swish = d_combined * self.act
        d_act = d_combined * (self.gate * self.swish_sig)
        d_gate = d_swish * (self.swish_sig * (1.0 + self.gate * (1.0 - self.swish_sig)))
        dz = np.hstack([d_gate, d_act])
        return self.w12.backward(dz)

    def params(self):
        return self.w12.params() + self.w3.params()

class MultiHeadAttention:
    def __init__(self, dim, heads=8):
        self.dim, self.heads = dim, heads
        self.hd = dim // heads
        self.scale = 1.0 / np.sqrt(self.hd)
        self.wqkv = Linear(dim, dim * 3)
        self.wo = Linear(dim, dim)
        self.q, self.k, self.v, self.att = [None] * 4

    def forward(self, x):
        b, d = x.shape
        qkv = self.wqkv.forward(x)
        qkv = qkv.reshape(b, 3, self.heads, self.hd)
        self.q, self.k, self.v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # (b, h, hd) @ (b, h, hd).T -> (b, h, h)
        dots = np.einsum('bhd,bmd->bhm', self.q, self.k) * self.scale
        self.att = stable_softmax(dots, axis=-1)
        
        out = np.einsum('bhm,bmd->bhd', self.att, self.v)
        self.context = out.reshape(b, d)
        return self.wo.forward(self.context)

    def backward(self, dout):
        b, d = dout.shape
        dctx = self.wo.backward(dout).reshape(b, self.heads, self.hd)
        
        datt = np.einsum('bhd,bmd->bhm', dctx, self.v)
        dv = np.einsum('bhm,bhd->bmd', self.att, dctx)
        
        # Softmax backward
        ds = self.att * (datt - np.sum(self.att * datt, axis=-1, keepdims=True))
        ds *= self.scale
        
        dq = np.einsum('bhm,bmd->bhd', ds, self.k)
        dk = np.einsum('bhm,bhd->bmd', ds, self.q)
        
        dqkv = np.stack([dq, dk, dv], axis=1).reshape(b, 3 * d)
        return self.wqkv.backward(dqkv)

    def params(self):
        return self.wqkv.params() + self.wo.params()

class RedundantMoE:
    def __init__(self, dim):
        # Gemini Expert: Deeper/Wider
        self.gemini = SwiGLU(dim, dim * 4)
        # Groq Expert: Fast/Lean
        self.groq = SwiGLU(dim, dim * 2)
        self.gate = Linear(dim, 2)
        self.p, self.o1, self.o2 = None, None, None

    def forward(self, x):
        g = self.gate.forward(x)
        self.p = stable_softmax(g, axis=-1)
        self.o1 = self.gemini.forward(x)
        self.o2 = self.groq.forward(x)
        return self.p[:, 0:1] * self.o1 + self.p[:, 1:2] * self.o2

    def backward(self, dout):
        p1, p2 = self.p[:, 0:1], self.p[:, 1:2]
        dx1 = self.gemini.backward(dout * p1)
        dx2 = self.groq.backward(dout * p2)
        
        dp1 = np.sum(dout * self.o1, axis=-1, keepdims=True)
        dp2 = np.sum(dout * self.o2, axis=-1, keepdims=True)
        
        dp = np.hstack([dp1, dp2])
        dg = self.p * (dp - np.sum(self.p * dp, axis=-1, keepdims=True))
        return dx1 + dx2 + self.gate.backward(dg)

    def params(self):
        return self.gemini.params() + self.groq.params() + self.gate.params()

class SovereignBlock:
    def __init__(self, dim):
        self.ln1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim)
        self.ln2 = RMSNorm(dim)
        self.moe = RedundantMoE(dim)

    def forward(self, x):
        x = x + self.attn.forward(self.ln1.forward(x))
        x = x + self.moe.forward(self.ln2.forward(x))
        return x

    def backward(self, dout):
        dm = self.moe.backward(dout)
        dn2 = self.ln2.backward(dm)
        dx_mid = dout + dn2
        da = self.attn.backward(self.ln1.forward(dx_mid)) # Approximation for speed
        # Correct residual backprop
        da = self.attn.backward(self.ln1.forward(dx_mid)) 
        # Re-evaluating for precision
        dn1 = self.ln1.backward(self.attn.backward(dout)) # Simplified path
        # Full path
        d_moe_in = self.moe.backward(dout)
        d_res2 = dout + self.ln2.backward(d_moe_in)
        d_attn_in = self.attn.backward(self.ln1.forward(d_res2))
        d_res1 = d_res2 + self.ln1.backward(d_attn_in)
        return d_res1

    def params(self):
        return self.ln1.params() + self.attn.params() + self.ln2.params() + self.moe.params()

class SovereignArchitect:
    def __init__(self, in_d, h_d, out_d, depth=4):
        self.stem = Linear(in_d, h_d)
        self.blocks = [SovereignBlock(h_d) for _ in range(depth)]
        self.norm = RMSNorm(h_d)
        self.head = Linear(h_d, out_d, init_scale=0.1)

    def forward(self, x):
        x = self.stem.forward(x)
        for b in self.blocks:
            x = b.forward(x)
        return self.head.forward(self.norm.forward(x))

    def backward(self, dout):
        dout = self.norm.backward(self.head.backward(dout))
        for b in reversed(self.blocks):
            dout = b.backward(dout)
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
            w, g = p["ref"], p["grad"]
            if self.wd > 0: w -= lr * self.wd * w
            update = np.sign(self.b1 * self.m[i] + (1.0 - self.b1) * g)
            w -= lr * update
            self.m[i] = self.b2 * self.m[i] + (1.0 - self.b2) * g

def get_data(n, d, k):
    X = np.random.randn(n, d).astype(np.float32)
    y = np.random.randint(0, k, n)
    centers = np.random.randn(k, d).astype(np.float32) * 5.0
    X += centers[y]
    X = (X - np.mean(X)) / (np.std(X) + 1e-6)
    return X, y

def train():
    N, D, K = 10000, 784, 10
    X, y = get_data(N, D, K)
    model = SovereignArchitect(D, 128, K, depth=3)
    opt = Lion(model.params(), lr=1e-4, wd=0.01)
    bs, epochs = 64, 30
    
    print(f"OMEGA-ASI | SOVEREIGN-V17 | PARAMS: {sum(p['ref'].size for p in model.params())}")
    
    for ep in range(epochs):
        idx = np.random.permutation(N)
        l_acc, a_acc, t0 = 0, 0, time.time()
        # Cosine Decay
        sched = 0.5 * (1 + np.cos(np.pi * ep / epochs))
        
        for i in range(0, N, bs):
            bi = idx[i : i + bs]
            xb, yb = X[bi], y[bi]
            m = xb.shape[0]
            
            logits = model.forward(xb)
            probs = stable_softmax(logits)
            
            loss = -np.mean(np.log(probs[range(m), yb] + 1e-10))
            l_acc += loss * (m / N)
            a_acc += np.mean(np.argmax(probs, axis=1) == yb) * (m / N)
            
            dout = probs.copy()
            dout[range(m), yb] -= 1
            model.backward(dout / m)
            
            # Global Gradient Clipping
            gn = np.sqrt(sum(np.sum(p["grad"]**2) for p in model.params() if p["grad"] is not None))
            if gn > 1.0:
                for p in model.params():
                    if p["grad"] is not None: p["grad"] /= (gn + 1e-6)
            
            opt.step(scale=sched)
            
        dt = time.time() - t0
        print(f"EP:{ep:02d} | LOSS:{l_acc:.4f} | ACC:{a_acc:.4f} | SPEED:{N/dt:.0f}s/s | LR:{opt.lr*sched:.7f}")

if __name__ == "__main__":
    train()
