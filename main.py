
import numpy as np
import time

class FastGELU:
    def forward(self, x):
        return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))

    def backward(self, x, dout):
        return dout * (0.5 * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x**3))) + 0.5 * x * (1 - np.tanh(0.7978845608 * (x + 0.044715 * x**3))**2) * 0.7978845608 * (1 + 3 * 0.044715 * x**2))

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

    def backward(self, x, dout):
        dx_hat = dout * self.gamma
        dgamma = np.sum(dout * self.x_hat, axis=0)
        dbeta = np.sum(dout, axis=0)
        dx = (1.0 / x.shape[-1]) * self.std_inv * (x.shape[-1] * dx_hat - np.sum(dx_hat, axis=-1, keepdims=True) - self.x_hat * np.sum(dx_hat * self.x_hat, axis=-1, keepdims=True))
        return dx, dgamma, dbeta

class Swish:
    def forward(self, x):
        return x * (1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))

    def backward(self, x, dout):
        return dout * (self.forward(x) + x * (1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))) * np.exp(-np.clip(x, -20, 20)))

class Linear:
    def __init__(self, in_d, out_d):
        scale = np.sqrt(2.0 / in_d)
        self.W = (np.random.randn(in_d, out_d) * scale).astype(np.float32)
        self.b = np.zeros(out_d, dtype=np.float32)

    def forward(self, x):
        return np.dot(x, self.W) + self.b

    def backward(self, x, dout):
        dW = np.dot(x.T, dout)
        db = np.sum(dout, axis=0)
        return np.dot(dout, self.W.T), dW, db

class RedundantEngine:
    def __init__(self, dim):
        self.gemini_path = Linear(dim, dim)
        self.groq_path = Linear(dim, dim)
        self.alpha = 0.5

    def forward(self, x):
        return self.alpha * self.gemini_path.forward(x) + (1 - self.alpha) * self.groq_path.forward(x)

    def backward(self, x, dout):
        d_gemini = self.gemini_path.backward(x, dout * self.alpha)
        d_groq = self.groq_path.backward(x, dout * (1 - self.alpha))
        return d_gemini[0] + d_groq[0], d_gemini[1], d_gemini[2], d_groq[1], d_groq[2]

class SovereignBlock:
    def __init__(self, dim, expansion=4):
        self.ln1 = LayerNorm(dim)
        self.engine = RedundantEngine(dim)
        self.ln2 = LayerNorm(dim)
        self.l1 = Linear(dim, dim * expansion)
        self.act = FastGELU()
        self.l2 = Linear(dim * expansion, dim)

    def forward(self, x):
        res = x
        h = self.ln1.forward(x)
        h = self.engine.forward(h)
        x = res + h

        res = x
        h = self.ln2.forward(x)
        h = self.l1.forward(h)
        h = self.act.forward(h)
        h = self.l2.forward(h)
        return x + h

    def backward(self, x, dout):
        res_dout = dout
        dh = self.l2.backward(self.act.forward(self.l1.forward(self.ln2.forward(x))), dout)
        dh = self.act.backward(self.l1.forward(self.ln2.forward(x)), dh)
        dh = self.l1.backward(self.ln2.forward(x), dh)
        dh = self.ln2.backward(x, dh)
        dout = res_dout + dh

        res_dout = dout
        dh = self.engine.backward(self.ln1.forward(x), dout)
        dh = self.ln1.backward(x, dh)
        return res_dout + dh

class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=0.01):
        self.lr, self.beta1, self.beta2, self.eps, self.wd = lr, betas[0], betas[1], eps, wd
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for i in range(len(params)):
            params[i] -= self.wd * params[i]
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i]**2)
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class SovereignArchitect:
    def __init__(self, in_d=784, h_d=256, out_d=10, depth=2):
        self.stem = Linear(in_d, h_d)
        self.blocks = [SovereignBlock(h_d) for _ in range(depth)]
        self.head_ln = LayerNorm(h_d)
        self.head = Linear(h_d, out_d)

        self.params = []
        for l in [self.stem] + self.blocks + [self.head_ln, self.head]:
            if hasattr(l, 'W'):
                self.params.extend([l.W, l.b])
            elif hasattr(l, 'gamma'):
                self.params.extend([l.gamma, l.beta])

        self.optimizer = AdamW(self.params)

    def forward(self, x):
        x = self.stem.forward(x)
        for block in self.blocks:
            x = block.forward(x)
        x = self.head_ln.forward(x)
        return self.head.forward(x)

    def backward(self, x, dout):
        dx = self.head.backward(x, dout)
        dx = self.head_ln.backward(x, dx)
        for block in reversed(self.blocks):
            dx = block.backward(x, dx)
        dx = self.stem.backward(x, dx)
        return dx

def run_evolution():
    N, D, K = 10000, 784, 10
    X = np.random.randn(N, D).astype(np.float32)
    y_true = np.random.randint(0, K, N)

    model = SovereignArchitect(in_d=D, h_d=128, out_d=K, depth=2)
    batch_size = 256
    epochs = 40

    print("OMEGA-ASI | RECURSIVE SELF-EVOLUTION START")
    start_time = time.time()

    for epoch in range(epochs):
        indices = np.random.permutation(N)
        epoch_loss, epoch_acc = 0, 0

        for i in range(0, N, batch_size):
            idx = indices[i:i+batch_size]
            xb, yb = X[idx], y_true[idx]
            m = xb.shape[0]

            logits = model.forward(xb)
            shift_logits = logits - np.max(logits, axis=1, keepdims=True)
            exps = np.exp(shift_logits)
            probs = exps / np.sum(exps, axis=1, keepdims=True)

            loss = -np.mean(np.log(probs[range(m), yb] + 1e-10))
            acc = np.mean(np.argmax(probs, axis=1) == yb)

            d_logits = probs.copy()
            d_logits[range(m), yb] -= 1
            d_logits /= m

            dx = model.backward(xb, d_logits)
            grads = []
            for p, g in zip(model.params, [dx] + model.stem.dW + model.stem.db + [model.head_ln.dgamma, model.head_ln.dbeta] + model.head.dW + model.head.db):
                grads.append(g)

            model.optimizer.step(model.params, grads)

            epoch_loss += loss * (m / N)
            epoch_acc += acc * (m / N)

        if epoch % 5 == 0:
            print(f"STEP:{epoch:03d} | LOSS:{epoch_loss:.4f} | ACC:{epoch_acc:.4f}")

    end_time = time.time()
    print(f"EVOLUTION_COMPLETE | TOTAL_TIME:{end_time - start_time:.2f}s | FINAL_ACC:{epoch_acc:.4f}")

if __name__ == "__main__":
    run_evolution()
