import numpy as np
import time

class FastGELU:
    def forward(self, x):
        self.x = x
        self.tanh_in = 0.7978845608 * (x + 0.044715 * x**3)
        self.t = np.tanh(self.tanh_in)
        return 0.5 * x * (1 + self.t)

    def backward(self, dout):
        sech2 = 1.0 - self.t**2
        grad = 0.5 * (1 + self.t) + (0.5 * self.x * sech2 * 0.7978845608 * (1 + 3 * 0.044715 * self.x**2))
        return dout * grad

class LayerNorm:
    def __init__(self, dim, eps=1e-6):
        self.gamma = np.ones(dim, dtype=np.float32)
        self.beta = np.zeros(dim, dtype=np.float32)
        self.eps = eps

    def forward(self, x):
        self.x = x
        self.mu = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.std_inv = 1.0 / np.sqrt(self.var + self.eps)
        self.x_hat = (x - self.mu) * self.std_inv
        return self.gamma * self.x_hat + self.beta

    def backward(self, dout):
        B, D = dout.shape
        self.dgamma = np.sum(dout * self.x_hat, axis=0)
        self.dbeta = np.sum(dout, axis=0)
        dx_hat = dout * self.gamma
        dx = (1.0 / D) * self.std_inv * (D * dx_hat - np.sum(dx_hat, axis=-1, keepdims=True) - self.x_hat * np.sum(dx_hat * self.x_hat, axis=-1, keepdims=True))
        return dx

class Linear:
    def __init__(self, in_d, out_d, kaiming=True):
        scale = np.sqrt(2.0 / in_d) if kaiming else 0.02
        self.W = (np.random.randn(in_d, out_d) * scale).astype(np.float32)
        self.b = np.zeros(out_d, dtype=np.float32)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0)
        return dout @ self.W.T

class GatedRedundantEngine:
    """Hybrid Architecture: Gemini (Contextual Stability) & Groq (High-Throughput Projection)"""
    def __init__(self, dim):
        self.gemini_path = Linear(dim, dim)
        self.groq_path = Linear(dim, dim)
        self.gate_layer = Linear(dim, 1)
        
    def forward(self, x):
        self.g_out = self.gemini_path.forward(x)
        self.q_out = self.groq_path.forward(x)
        self.gate_logit = self.gate_layer.forward(x)
        self.gate = 1.0 / (1.0 + np.exp(-self.gate_logit))
        return self.gate * self.g_out + (1.0 - self.gate) * self.q_out

    def backward(self, dout):
        dg_out = dout * self.gate
        dq_out = dout * (1.0 - self.gate)
        dgate = np.sum(dout * (self.g_out - self.q_out), axis=-1, keepdims=True)
        dgate_logit = dgate * self.gate * (1.0 - self.gate)
        
        dx_g = self.gemini_path.backward(dg_out)
        dx_q = self.groq_path.backward(dq_out)
        dx_gate = self.gate_layer.backward(dgate_logit)
        return dx_g + dx_q + dx_gate

    def get_params(self):
        return [self.gemini_path.W, self.gemini_path.b, self.groq_path.W, self.groq_path.b, self.gate_layer.W, self.gate_layer.b]

    def get_grads(self):
        return [self.gemini_path.dW, self.gemini_path.db, self.groq_path.dW, self.groq_path.db, self.gate_layer.dW, self.gate_layer.db]

class SovereignBlock:
    def __init__(self, dim, expansion=4):
        self.ln1 = LayerNorm(dim)
        self.engine = GatedRedundantEngine(dim)
        self.ln2 = LayerNorm(dim)
        self.l1 = Linear(dim, dim * expansion)
        self.act = FastGELU()
        self.l2 = Linear(dim * expansion, dim)

    def forward(self, x):
        # Pre-Norm Residual Path 1
        h = self.ln1.forward(x)
        h = self.engine.forward(h)
        x = x + h
        # Pre-Norm Residual Path 2
        h = self.ln2.forward(x)
        h = self.l1.forward(h)
        h = self.act.forward(h)
        h = self.l2.forward(h)
        self.out = x + h
        return self.out

    def backward(self, dout):
        dout_res2 = dout
        dh = self.l2.backward(dout)
        dh = self.act.backward(dh)
        dh = self.l1.backward(dh)
        dh = self.ln2.backward(dh)
        dout = dout_res2 + dh
        
        dout_res1 = dout
        dh = self.engine.backward(dout)
        dh = self.ln1.backward(dh)
        return dout_res1 + dh

    def get_params(self):
        p = self.engine.get_params()
        p.extend([self.l1.W, self.l1.b, self.l2.W, self.l2.b, self.ln1.gamma, self.ln1.beta, self.ln2.gamma, self.ln2.beta])
        return p

    def get_grads(self):
        g = self.engine.get_grads()
        g.extend([self.l1.dW, self.l1.db, self.l2.dW, self.l2.db, self.ln1.dgamma, self.ln1.dbeta, self.ln2.dgamma, self.ln2.dbeta])
        return g

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
        self.head_ln = LayerNorm(h_d)
        self.head = Linear(h_d, out_d)
        self.layers = [self.stem] + self.blocks + [self.head_ln, self.head]
        self.params = []
        for l in self.layers:
            if hasattr(l, 'get_params'): self.params.extend(l.get_params())
            elif hasattr(l, 'W'): self.params.extend([l.W, l.b])
            else: self.params.extend([l.gamma, l.beta])
        self.optimizer = AdamW(self.params, lr=1e-3, wd=0.05)

    def forward(self, x):
        for l in self.layers: x = l.forward(x)
        return x

    def backward(self, dout):
        for l in reversed(self.layers): dout = l.backward(dout)
        grads = []
        for l in self.layers:
            if hasattr(l, 'get_grads'): grads.extend(l.get_grads())
            elif hasattr(l, 'W'): grads.extend([l.dW, l.db])
            else: grads.extend([l.dgamma, l.dbeta])
        return grads

def run_evolution():
    N, D, K = 5000, 784, 10
    # Generate structured synthetic data
    X = np.random.randn(N, D).astype(np.float32)
    W_true = np.random.randn(D, K).astype(np.float32)
    y_true = np.argmax(X @ W_true + np.random.randn(N, K) * 0.1, axis=1)
    
    model = SovereignArchitect(in_d=D, h_d=128, out_d=K, depth=3)
    batch_size = 128
    epochs = 50

    print("OMEGA-ASI | RECURSIVE SELF-EVOLUTION INITIATED")
    start_time = time.time()

    for epoch in range(epochs):
        indices = np.random.permutation(N)
        epoch_loss, epoch_acc = 0, 0
        # Cosine Annealing
        lr_mult = 0.5 * (1 + np.cos(np.pi * epoch / epochs))

        for i in range(0, N, batch_size):
            idx = indices[i:i+batch_size]
            xb, yb = X[idx], y_true[idx]
            m = xb.shape[0]

            logits = model.forward(xb)
            probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs /= np.sum(probs, axis=1, keepdims=True)

            loss = -np.mean(np.log(probs[range(m), yb] + 1e-10))
            acc = np.mean(np.argmax(probs, axis=1) == yb)

            d_logits = probs.copy()
            d_logits[range(m), yb] -= 1
            d_logits /= m

            grads = model.backward(d_logits)
            
            # Global Gradient Clipping
            gnorm = np.sqrt(sum(np.sum(g**2) for g in grads))
            if gnorm > 1.0:
                for g in grads: g *= (1.0 / gnorm)
            
            model.optimizer.step(model.params, grads, lr_mult)

            epoch_loss += loss * (m / N)
            epoch_acc += acc * (m / N)

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"EPOCH:{epoch:03d} | LOSS:{epoch_loss:.4f} | ACC:{epoch_acc:.4f} | LR_M:{lr_mult:.3f}")

    total_time = time.time() - start_time
    print(f"EVOLUTION_COMPLETE | TIME:{total_time:.2f}s | FINAL_ACC:{epoch_acc:.4f}")

if __name__ == "__main__":
    run_evolution()
