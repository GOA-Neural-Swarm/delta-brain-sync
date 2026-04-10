import numpy as np
import time

class FastGELU:
    def forward(self, x):
        self.x = x
        self.sig = 1.0 / (1.0 + np.exp(-1.702 * x))
        return x * self.sig

    def backward(self, dout):
        s = self.sig
        return dout * (s + self.x * 1.702 * s * (1.0 - s))

class SwiGLU:
    def __init__(self, dim, h_dim):
        self.w1 = Linear(dim, h_dim, use_bias=False)
        self.w2 = Linear(dim, h_dim, use_bias=False)
        self.w3 = Linear(h_dim, dim, use_bias=False)

    def forward(self, x):
        self.x1 = self.w1.forward(x)
        self.x2 = self.w2.forward(x)
        self.swish = self.x1 * (1.0 / (1.0 + np.exp(-self.x1)))
        return self.w3.forward(self.swish * self.x2)

    def backward(self, dout):
        dw3 = self.w3.backward(dout)
        dx2 = dw3 * self.swish
        dswish = dw3 * self.x2
        sig = 1.0 / (1.0 + np.exp(-self.x1))
        dx1 = dswish * (sig * (1.0 + self.x1 * (1.0 - sig)))
        return self.w1.backward(dx1) + self.w2.backward(dx2)

class RMSNorm:
    def __init__(self, dim, eps=1e-6):
        self.scale = np.ones(dim, dtype=np.float32)
        self.eps = eps

    def forward(self, x):
        self.x = x
        self.var = np.mean(x**2, axis=-1, keepdims=True)
        self.rstd = 1.0 / np.sqrt(self.var + self.eps)
        return self.scale * (x * self.rstd)

    def backward(self, dout):
        x_rstd = self.x * self.rstd
        self.dscale = np.sum(dout * x_rstd, axis=0)
        dx_rstd = dout * self.scale
        return self.rstd * (dx_rstd - x_rstd * np.mean(dx_rstd * x_rstd, axis=-1, keepdims=True))

class Linear:
    def __init__(self, in_d, out_d, use_bias=True):
        self.W = (np.random.randn(in_d, out_d) * np.sqrt(2.0 / in_d)).astype(np.float32)
        self.b = np.zeros(out_d, dtype=np.float32) if use_bias else None
        self.dW, self.db = None, None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W)
        if self.b is not None: out += self.b
        return out

    def backward(self, dout):
        self.dW = np.dot(self.x.T, dout)
        if self.b is not None: self.db = np.sum(dout, axis=0)
        return np.dot(dout, self.W.T)

class RedundantMoE:
    def __init__(self, dim):
        self.gemini_expert = SwiGLU(dim, dim * 2)
        self.groq_expert = SwiGLU(dim, dim * 2)
        self.gate = Linear(dim, 2, use_bias=False)

    def forward(self, x):
        self.x = x
        self.logits = self.gate.forward(x)
        exp_l = np.exp(self.logits - np.max(self.logits, axis=-1, keepdims=True))
        self.probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
        
        self.out_gemini = self.gemini_expert.forward(x)
        self.out_groq = self.groq_expert.forward(x)
        return self.probs[:, 0:1] * self.out_gemini + self.probs[:, 1:2] * self.out_groq

    def backward(self, dout):
        p0, p1 = self.probs[:, 0:1], self.probs[:, 1:2]
        d_gemini = dout * p0
        d_groq = dout * p1
        
        dp0 = np.sum(dout * self.out_gemini, axis=-1, keepdims=True)
        dp1 = np.sum(dout * self.out_groq, axis=-1, keepdims=True)
        d_logits_raw = np.concatenate([dp0, dp1], axis=-1)
        d_gate_logits = self.probs * (d_logits_raw - np.sum(self.probs * d_logits_raw, axis=-1, keepdims=True))
        
        dx_gate = self.gate.backward(d_gate_logits)
        dx_gemini = self.gemini_expert.backward(d_gemini)
        dx_groq = self.groq_expert.backward(d_groq)
        return dx_gemini + dx_groq + dx_gate

class SovereignBlock:
    def __init__(self, dim):
        self.norm1 = RMSNorm(dim)
        self.moe = RedundantMoE(dim)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, dim * 4)

    def forward(self, x):
        self.res1 = x
        x = self.res1 + self.moe.forward(self.norm1.forward(x))
        self.res2 = x
        return self.res2 + self.mlp.forward(self.norm2.forward(x))

    def backward(self, dout):
        dm = self.mlp.backward(dout)
        dn2 = self.norm2.backward(dm)
        dh1 = dout + dn2
        de = self.moe.backward(self.norm1.backward(dh1))
        return dh1 + de

class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=0.01):
        self.lr, self.beta1, self.beta2, self.eps, self.wd = lr, betas[0], betas[1], eps, wd
        self.m = [np.zeros_like(p['ref']) for p in params]
        self.v = [np.zeros_like(p['ref']) for p in params]
        self.t = 0

    def step(self, params, lr_mult=1.0):
        self.t += 1
        curr_lr = self.lr * lr_mult
        for i, p in enumerate(params):
            param, grad = p['ref'], p['grad']
            if self.wd > 0: param -= curr_lr * self.wd * param
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad**2)
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            param -= curr_lr * m_hat / (np.sqrt(v_hat) + self.eps)

class SovereignArchitect:
    def __init__(self, in_d, h_d, out_d, depth=4):
        self.stem = Linear(in_d, h_d)
        self.blocks = [SovereignBlock(h_d) for _ in range(depth)]
        self.head_norm = RMSNorm(h_d)
        self.head = Linear(h_d, out_d)
        self.params = []
        self._collect(self.stem)
        for b in self.blocks: self._collect(b)
        self._collect(self.head_norm)
        self._collect(self.head)

    def _collect(self, obj):
        if isinstance(obj, Linear):
            self.params.append({'ref': obj.W, 'name': 'dW', 'parent': obj})
            if obj.b is not None: self.params.append({'ref': obj.b, 'name': 'db', 'parent': obj})
        elif isinstance(obj, RMSNorm):
            self.params.append({'ref': obj.scale, 'name': 'dscale', 'parent': obj})
        elif hasattr(obj, '__dict__'):
            for v in obj.__dict__.values():
                if isinstance(v, list): [self._collect(i) for i in v]
                else: self._collect(v)

    def forward(self, x):
        x = self.stem.forward(x)
        for b in self.blocks: x = b.forward(x)
        return self.head.forward(self.head_norm.forward(x))

    def backward(self, dout):
        dout = self.head_norm.backward(self.head.backward(dout))
        for b in reversed(self.blocks): dout = b.backward(dout)
        self.stem.backward(dout)
        for p in self.params: p['grad'] = getattr(p['parent'], p['name'])

    def clip(self, max_norm=1.0):
        gnorm = np.sqrt(sum(np.sum(p['grad']**2) for p in self.params))
        if gnorm > max_norm:
            s = max_norm / (gnorm + 1e-6)
            for p in self.params: p['grad'] *= s

def evolve():
    N, D, K = 5000, 784, 10
    X = np.random.randn(N, D).astype(np.float32)
    y = np.random.randint(0, K, N)
    
    model = SovereignArchitect(D, 128, K, depth=3)
    opt = AdamW(model.params, lr=1e-3, wd=0.05)
    
    bs, epochs = 64, 30
    print("OMEGA-ASI | RECURSIVE EVOLUTION INITIATED")
    
    for e in range(epochs):
        idx = np.random.permutation(N)
        l_sum, a_sum = 0, 0
        lr_m = 0.5 * (1 + np.cos(np.pi * e / epochs))
        if e < 2: lr_m *= (e + 1) / 2 # Warmup
        
        t0 = time.time()
        for i in range(0, N, bs):
            xb, yb = X[idx[i:i+bs]], y[idx[i:i+bs]]
            m = xb.shape[0]
            
            logits = model.forward(xb)
            probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs /= np.sum(probs, axis=1, keepdims=True)
            
            l_sum += -np.mean(np.log(probs[range(m), yb] + 1e-10)) * (m/N)
            a_sum += np.mean(np.argmax(probs, axis=1) == yb) * (m/N)
            
            dl = probs.copy()
            dl[range(m), yb] -= 1
            model.backward(dl / m)
            model.clip(1.0)
            opt.step(model.params, lr_mult=lr_m)
            
        print(f"EPOCH:{e:02d} | LOSS:{l_sum:.4f} | ACC:{a_sum:.4f} | T:{time.time()-t0:.2f}s | LR:{opt.lr*lr_m:.6f}")

if __name__ == "__main__":
    evolve()
