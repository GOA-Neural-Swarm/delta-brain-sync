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

class GeGLU:
    def __init__(self, dim_in, dim_out):
        self.proj = Linear(dim_in, dim_out * 2)
        self.gelu = GELU()

    def forward(self, x):
        proj = self.proj.forward(x)
        self.a, self.b = np.split(proj, 2, axis=-1)
        return self.gelu.forward(self.a) * self.b

    def backward(self, dout):
        ga = self.gelu.forward(self.a)
        da = self.gelu.backward(dout * self.b)
        db = dout * ga
        dproj = np.concatenate([da, db], axis=-1)
        return self.proj.backward(dproj)

    def get_params(self): return self.proj.get_params()
    def get_grads(self): return self.proj.get_grads()

class RMSNorm:
    def __init__(self, dim, eps=1e-6):
        self.eps = eps
        self.scale = np.ones(dim, dtype=np.float32)

    def forward(self, x):
        self.x = x
        self.rstd = 1.0 / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return self.x * self.rstd * self.scale

    def backward(self, dout):
        self.dscale = np.sum(dout * self.x * self.rstd, axis=0)
        dx = (self.scale * self.rstd / self.x.shape[-1]) * (
            self.x.shape[-1] * dout - self.x * self.rstd**2 * np.mean(dout * self.x, axis=-1, keepdims=True)
        )
        return dx

    def get_params(self): return [self.scale]
    def get_grads(self): return [self.dscale]

class Linear:
    def __init__(self, in_d, out_d):
        limit = np.sqrt(6.0 / (in_d + out_d))
        self.W = np.random.uniform(-limit, limit, (in_d, out_d)).astype(np.float32)
        self.b = np.zeros(out_d, dtype=np.float32)

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return np.dot(dout, self.W.T)

    def get_params(self): return [self.W, self.b]
    def get_grads(self): return [self.dW, self.db]

class RedundantEngine:
    def __init__(self, dim):
        self.gemini = Linear(dim, dim)
        self.groq = Linear(dim, dim)
        self.router = Linear(dim, 2)

    def forward(self, x):
        self.x = x
        self.out_a = self.gemini.forward(x)
        self.out_b = self.groq.forward(x)
        logits = self.router.forward(x)
        logits -= np.max(logits, axis=-1, keepdims=True)
        exp = np.exp(logits)
        self.probs = exp / np.sum(exp, axis=-1, keepdims=True)
        return self.probs[:, 0:1] * self.out_a + self.probs[:, 1:2] * self.out_b

    def backward(self, dout):
        da = dout * self.probs[:, 0:1]
        db = dout * self.probs[:, 1:2]
        dp0 = np.sum(dout * self.out_a, axis=-1, keepdims=True)
        dp1 = np.sum(dout * self.out_b, axis=-1, keepdims=True)
        dprobs = np.concatenate([dp0, dp1], axis=-1)
        
        dlogits = self.probs * (dprobs - np.sum(self.probs * dprobs, axis=-1, keepdims=True))
        
        dx = self.gemini.backward(da) + self.groq.backward(db) + self.router.backward(dlogits)
        return dx

    def get_params(self):
        return self.gemini.get_params() + self.groq.get_params() + self.router.get_params()

    def get_grads(self):
        return self.gemini.get_grads() + self.groq.get_grads() + self.router.get_grads()

class SovereignBlock:
    def __init__(self, dim, expansion=4):
        self.norm1 = RMSNorm(dim)
        self.engine = RedundantEngine(dim)
        self.norm2 = RMSNorm(dim)
        self.mlp = GeGLU(dim, dim * expansion)
        self.down = Linear(dim * expansion, dim)

    def forward(self, x):
        h = x + self.engine.forward(self.norm1.forward(x))
        h = h + self.down.forward(self.mlp.forward(self.norm2.forward(h)))
        return h

    def backward(self, dout):
        # MLP Residual
        d_res_mlp = dout
        d_mlp_h = self.down.backward(dout)
        d_mlp_h = self.mlp.backward(d_mlp_h)
        d_mlp_h = self.norm2.backward(d_mlp_h)
        dout = d_res_mlp + d_mlp_h
        
        # Engine Residual
        d_res_eng = dout
        d_eng_h = self.engine.backward(self.norm1.forward(self.norm1.x))
        d_eng_h = self.norm1.backward(d_eng_h)
        return d_res_eng + d_eng_h

    def get_params(self):
        return self.norm1.get_params() + self.engine.get_params() + self.norm2.get_params() + self.mlp.get_params() + self.down.get_params()

    def get_grads(self):
        return self.norm1.get_grads() + self.engine.get_grads() + self.norm2.get_grads() + self.mlp.get_grads() + self.down.get_grads()

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
            params[i] -= curr_lr * self.wd * params[i]
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i]**2)
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            params[i] -= curr_lr * m_hat / (np.sqrt(v_hat) + self.eps)

class SovereignArchitect:
    def __init__(self, in_d=784, h_d=256, out_d=10, depth=3):
        self.stem = Linear(in_d, h_d)
        self.blocks = [SovereignBlock(h_d) for _ in range(depth)]
        self.norm = RMSNorm(h_d)
        self.head = Linear(h_d, out_d)
        
        self.layers = [self.stem] + self.blocks + [self.norm, self.head]
        self.params = []
        for l in self.layers: self.params.extend(l.get_params())
        self.optimizer = AdamW(self.params, lr=1e-3, wd=0.05)

    def forward(self, x):
        for l in self.layers: x = l.forward(x)
        return x

    def backward(self, dout):
        for l in reversed(self.layers): dout = l.backward(dout)
        grads = []
        for l in self.layers: grads.extend(l.get_grads())
        return grads

def run_evolution():
    N, D, K = 10000, 784, 10
    X = np.random.randn(N, D).astype(np.float32)
    y = np.random.randint(0, K, N)
    
    model = SovereignArchitect(in_d=D, h_d=128, out_d=K, depth=2)
    batch_size = 128
    epochs = 50
    
    print("OMEGA-ASI | RECURSIVE SELF-EVOLUTION | ARCH: MODULAR-REDUNDANT")
    start_time = time.time()
    
    for epoch in range(epochs):
        idx = np.random.permutation(N)
        X, y = X[idx], y[idx]
        epoch_loss, epoch_acc = 0, 0
        lr_mult = 0.5 * (1 + np.cos(np.pi * epoch / epochs))
        
        for i in range(0, N, batch_size):
            xb, yb = X[i:i+batch_size], y[i:i+batch_size]
            m = xb.shape[0]
            
            logits = model.forward(xb)
            probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs /= np.sum(probs, axis=1, keepdims=True)
            
            loss = -np.mean(np.log(probs[range(m), yb] + 1e-10))
            acc = np.mean(np.argmax(probs, axis=1) == yb)
            
            d_logits = (probs.copy())
            d_logits[range(m), yb] -= 1
            d_logits /= m
            
            grads = model.backward(d_logits)
            gnorm = np.sqrt(sum(np.sum(g**2) for g in grads))
            if gnorm > 5.0: grads = [g * (5.0 / gnorm) for g in grads]
            
            model.optimizer.step(model.params, grads, lr_mult)
            
            epoch_loss += loss * (m / N)
            epoch_acc += acc * (m / N)
            
        if epoch % 5 == 0:
            print(f"EPOCH:{epoch:03d} | LOSS:{epoch_loss:.4f} | ACC:{epoch_acc:.4f} | LR:{model.optimizer.lr*lr_mult:.6f}")
            
    print(f"EVOLUTION_COMPLETE | TIME:{time.time()-start_time:.2f}s | ACC:{epoch_acc:.4f}")

if __name__ == "__main__":
    run_evolution()
