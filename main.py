
import numpy as np
import time

class GELU:
    def forward(self, x):
        self.x = x
        self.tanh = np.tanh(0.7978845608 * (x + 0.044715 * x**3))
        return 0.5 * x * (1 + self.tanh)

    def backward(self, dout):
        x = self.x
        sech2 = 1.0 - self.tanh**2
        grad = 0.5 * (1 + self.tanh) + (0.5 * x * sech2 * 0.7978845608 * (1 + 3 * 0.044715 * x**2))
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
        limit = np.sqrt(6.0 / (in_d + out_d))
        self.W = np.random.uniform(-limit, limit, (in_d, out_d)).astype(np.float32)
        self.b = np.zeros(out_d, dtype=np.float32) if use_bias else None

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

    def forward(self, x):
        self.x = x
        self.out_gemini = self.gemini_path.forward(x)
        self.out_groq = self.groq_path.forward(x)

        g = self.gate.forward(x)
        exp_g = np.exp(g - np.max(g, axis=-1, keepdims=True))
        self.probs = exp_g / np.sum(exp_g, axis=-1, keepdims=True)

        return self.probs[:, 0:1] * self.out_gemini + self.probs[:, 1:2] * self.out_groq

    def backward(self, dout):
        d_gemini = dout * self.probs[:, 0:1]
        d_groq = dout * self.probs[:, 1:2]

        d_p0 = np.sum(dout * self.out_gemini, axis=-1, keepdims=True)
        d_p1 = np.sum(dout * self.out_groq, axis=-1, keepdims=True)
        d_probs_raw = np.concatenate([d_p0, d_p1], axis=-1)

        d_gate_logits = self.probs * (d_probs_raw - np.sum(self.probs * d_probs_raw, axis=-1, keepdims=True))

        dx_gate = self.gate.backward(d_gate_logits)
        dx_gemini = self.gemini_path.backward(d_gemini)
        dx_groq = self.groq_path.backward(d_groq)

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
        h = x + self.engine.forward(self.norm1.forward(x))
        h = h + self.mlp_out.forward(self.act.forward(self.mlp_in.forward(self.norm2.forward(h))))
        return h

    def backward(self, dout):
        res_mlp = dout
        dm = self.mlp_out.backward(dout)
        dm = self.act.backward(dm)
        dm = self.mlp_in.backward(dm)
        dm = self.norm2.backward(dm)
        dout = res_mlp + dm

        res_eng = dout
        de = self.engine.backward(dout)
        de = self.norm1.backward(de)
        return res_eng + de

class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=0.01):
        self.lr, self.beta1, self.beta2, self.eps, self.wd = lr, betas[0], betas[1], eps, wd
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads, lr_mult=1.0):
        self.t += 1
        curr_lr = self.lr * lr_mult
        for i in range(len(params)):
            if self.wd > 0: params[i] -= curr_lr * self.wd * params[i]
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i]**2)
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            params[i] -= curr_lr * m_hat / (np.sqrt(v_hat) + self.eps)

class SovereignArchitect:
    def __init__(self, in_d, h_d, out_d, depth=2):
        self.stem = Linear(in_d, h_d)
        self.blocks = [SovereignBlock(h_d) for _ in range(depth)]
        self.head_norm = RMSNorm(h_d)
        self.head = Linear(h_d, out_d)

        self.layers = [self.stem] + self.blocks + [self.head_norm, self.head]
        self.params, self.param_refs = [], []
        self._collect_params()
        self.optimizer = AdamW(self.params, lr=1e-3, wd=0.01)

    def _collect_params(self):
        def walk(obj):
            if hasattr(obj, 'W'): 
                self.params.append(obj.W)
                if obj.b is not None: self.params.append(obj.b)
            if hasattr(obj, 'scale'): self.params.append(obj.scale)
            if hasattr(obj, '__dict__'):
                for v in obj.__dict__.values():
                    if isinstance(v, (Linear, RMSNorm, RedundantEngine, SovereignBlock)): walk(v)
                    elif isinstance(v, list): [walk(i) for i in v]
        for l in self.layers: walk(l)

    def forward(self, x):
        for l in self.layers: x = l.forward(x)
        return x

    def backward(self, dout):
        for l in reversed(self.layers): dout = l.backward(dout)
        grads = []
        def walk_grads(obj):
            if hasattr(obj, 'dW'): 
                grads.append(obj.dW)
                if obj.b is not None: grads.append(obj.db)
            if hasattr(obj, 'dscale'): grads.append(obj.dscale)
            if hasattr(obj, '__dict__'):
                for v in obj.__dict__.values():
                    if isinstance(v, (Linear, RMSNorm, RedundantEngine, SovereignBlock)): walk_grads(v)
                    elif isinstance(v, list): [walk_grads(i) for i in v]
        for l in self.layers: walk_grads(l)
        return grads

def run_evolution():
    N, D, K = 5000, 784, 10
    X = np.random.randn(N, D).astype(np.float32)
    y = np.random.randint(0, K, N)

    model = SovereignArchitect(D, 128, K, depth=4)
    bs, epochs = 128, 50

    print("OMEGA-ASI | ARCHITECTURE INITIALIZED")
    for epoch in range(epochs):
        idx = np.random.permutation(N)
        L, A = 0, 0
        lr_m = 0.5 * (1 + np.cos(np.pi * epoch / epochs))

        for i in range(0, N, bs):
            xb, yb = X[idx[i:i+bs]], y[idx[i:i+bs]]
            m = xb.shape[0]

            logits = model.forward(xb)
            probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs /= np.sum(probs, axis=1, keepdims=True)

            L += -np.sum(np.log(probs[range(m), yb] + 1e-10)) / N
            A += np.sum(np.argmax(probs, axis=1) == yb) / N

            d_logits = probs.copy()
            d_logits[range(m), yb] -= 1
            d_logits /= m

            grads = model.backward(d_logits)
            gnorm = np.sqrt(sum(np.sum(g**2) for g in grads))
            if gnorm > 1.0: grads = [g / gnorm for g in grads]

            model.optimizer.step(model.params, grads, lr_m)

        if epoch % 10 == 0:
            print(f"EPOCH:{epoch:03d} | LOSS:{L:.4f} | ACC:{A:.4f} | LR:{model.optimizer.lr*lr_m:.5f}")

if __name__ == "__main__":
    run_evolution()
