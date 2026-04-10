import numpy as np
import time
import omega_point

class GELU:
    def forward(self, x):
        self.x = x
        self.tanh = np.tanh(0.7978845608 * (x + 0.044715 * x**3))
        return 0.5 * x * (1 + self.tanh)

    def backward(self, dout):
        sech2 = 1.0 - self.tanh**2
        grad = 0.5 * (1 + self.tanh) + (0.5 * self.x * sech2 * 0.7978845608 * (1 + 3 * 0.044715 * self.x**2))
        return dout * grad

class RMSNorm:
    def __init__(self, dim, eps=1e-6):
        self.scale = np.ones(dim, dtype=np.float32)
        self.eps = eps

    def forward(self, x):
        self.x = x
        self.rstd = 1.0 / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return self.scale * (x * self.rstd)

    def backward(self, dout):
        x_scaled = self.x * self.rstd
        self.dscale = np.sum(dout * x_scaled, axis=0)
        dx_scaled = dout * self.scale
        return self.rstd * (dx_scaled - x_scaled * np.mean(dx_scaled * x_scaled, axis=-1, keepdims=True))

class Linear:
    def __init__(self, in_d, out_d, use_bias=True):
        limit = np.sqrt(2.0 / in_d)
        self.W = np.random.normal(0, limit, (in_d, out_d)).astype(np.float32)
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

class RedundantEngine:
    def __init__(self, dim):
        self.gemini_path = Linear(dim, dim)
        self.groq_path = Linear(dim, dim)
        self.gate = Linear(dim, 2)
        self.act_gemini = GELU()
        self.act_groq = GELU()

    def forward(self, x):
        self.x = x
        self.out_gemini = self.act_gemini.forward(self.gemini_path.forward(x))
        self.out_groq = self.act_groq.forward(self.groq_path.forward(x))
        
        g = self.gate.forward(x)
        g_exp = np.exp(g - np.max(g, axis=-1, keepdims=True))
        self.probs = g_exp / np.sum(g_exp, axis=-1, keepdims=True)
        
        return self.probs[:, 0:1] * self.out_gemini + self.probs[:, 1:2] * self.out_groq

    def backward(self, dout):
        d_gemini = dout * self.probs[:, 0:1]
        d_groq = dout * self.probs[:, 1:2]
        
        d_p0 = np.sum(dout * self.out_gemini, axis=-1, keepdims=True)
        d_p1 = np.sum(dout * self.out_groq, axis=-1, keepdims=True)
        d_probs_raw = np.concatenate([d_p0, d_p1], axis=-1)
        
        d_gate_logits = self.probs * (d_probs_raw - np.sum(self.probs * d_probs_raw, axis=-1, keepdims=True))
        
        dx_gate = self.gate.backward(d_gate_logits)
        dx_gemini = self.gemini_path.backward(self.act_gemini.backward(d_gemini))
        dx_groq = self.groq_path.backward(self.act_groq.backward(d_groq))
        
        return dx_gemini + dx_groq + dx_gate

class SovereignBlock:
    def __init__(self, dim, expansion=4):
        self.norm1 = RMSNorm(dim)
        self.engine = RedundantEngine(dim)
        self.norm2 = RMSNorm(dim)
        self.mlp_in = Linear(dim, dim * expansion)
        self.act = GELU()
        self.mlp_out = Linear(dim * expansion, dim)

    def forward(self, x):
        self.res1 = x
        h = self.engine.forward(self.norm1.forward(x))
        self.h1 = self.res1 + h
        self.res2 = self.h1
        m = self.mlp_out.forward(self.act.forward(self.mlp_in.forward(self.norm2.forward(self.h1))))
        return self.res2 + m

    def backward(self, dout):
        dm = self.mlp_out.backward(dout)
        dm = self.act.backward(dm)
        dm = self.mlp_in.backward(dm)
        dm = self.norm2.backward(dm)
        dh1 = dout + dm
        
        de = self.engine.backward(dh1)
        de = self.norm1.backward(de)
        return dh1 + de

class AdamW:
    def __init__(self, params_meta, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, wd=0.01):
        self.lr, self.beta1, self.beta2, self.eps, self.wd = lr, betas[0], betas[1], eps, wd
        self.m = [np.zeros_like(p['ref']) for p in params_meta]
        self.v = [np.zeros_like(p['ref']) for p in params_meta]
        self.t = 0

    def step(self, params_meta, lr_mult=1.0):
        self.t += 1
        curr_lr = self.lr * lr_mult
        for i, p in enumerate(params_meta):
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
        self.params_meta = []
        self._map_params()

    def _map_params(self):
        def walk(obj):
            if isinstance(obj, Linear):
                self.params_meta.append({'ref': obj.W, 'grad_name': 'dW', 'parent': obj})
                if obj.b is not None: self.params_meta.append({'ref': obj.b, 'grad_name': 'db', 'parent': obj})
            elif isinstance(obj, RMSNorm):
                self.params_meta.append({'ref': obj.scale, 'grad_name': 'dscale', 'parent': obj})
            elif hasattr(obj, '__dict__'):
                for v in obj.__dict__.values():
                    if isinstance(v, list): [walk(i) for i in v]
                    else: walk(v)
        walk(self.stem)
        for b in self.blocks: walk(b)
        walk(self.head_norm)
        walk(self.head)

    def forward(self, x):
        x = self.stem.forward(x)
        for b in self.blocks: x = b.forward(x)
        x = self.head_norm.forward(x)
        return self.head.forward(x)

    def backward(self, dout):
        dout = self.head.backward(dout)
        dout = self.head_norm.backward(dout)
        for b in reversed(self.blocks): dout = b.backward(dout)
        self.stem.backward(dout)
        for p in self.params_meta:
            p['grad'] = getattr(p['parent'], p['grad_name'])

    def clip_grads(self, max_norm=1.0):
        gnorm = np.sqrt(sum(np.sum(p['grad']**2) for p in self.params_meta))
        if gnorm > max_norm:
            scale = max_norm / (gnorm + 1e-6)
            for p in self.params_meta: p['grad'] *= scale

def run_evolution():
    N, D, K = 10000, 784, 10
    X = np.random.randn(N, D).astype(np.float32)
    y = np.random.randint(0, K, N)

    model = SovereignArchitect(D, 256, K, depth=4)
    optimizer = AdamW(model.params_meta, lr=2e-3, wd=0.02)

    bs, epochs = 128, 20
    print(f"OMEGA-ASI | ARCHITECTURE: SOVEREIGN | STATUS: EVOLVING")

    for epoch in range(epochs):
        idx = np.random.permutation(N)
        total_loss, total_acc = 0, 0
        lr_m = 0.5 * (1 + np.cos(np.pi * epoch / epochs))
        
        start = time.time()
        for i in range(0, N, bs):
            batch_idx = idx[i:i+bs]
            xb, yb = X[batch_idx], y[batch_idx]
            m = xb.shape[0]

            logits = model.forward(xb)
            
            shift_logits = logits - np.max(logits, axis=1, keepdims=True)
            exps = np.exp(shift_logits)
            probs = exps / np.sum(exps, axis=1, keepdims=True)
            
            loss = -np.mean(np.log(probs[range(m), yb] + 1e-10))
            acc = np.mean(np.argmax(probs, axis=1) == yb)
            
            total_loss += loss * (m / N)
            total_acc += acc * (m / N)

            d_logits = probs.copy()
            d_logits[range(m), yb] -= 1
            d_logits /= m

            model.backward(d_logits)
            model.clip_grads(1.0)
            optimizer.step(model.params_meta, lr_mult=lr_m)

        end = time.time()
        print(f"E:{epoch:02d} | L:{total_loss:.4f} | A:{total_acc:.4f} | T:{end-start:.2f}s | LR:{optimizer.lr*lr_m:.5f}")

if __name__ == "__main__":
    run_evolution()
