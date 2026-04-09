
import numpy as np
import time

class FastGELU:
    def forward(self, x):
        self.x = x
        self.tanh_in = 0.7978845608 * (x + 0.044715 * x**3)
        self.t = np.tanh(self.tanh_in)
        return 0.5 * x * (1 + self.t)

    def backward(self, dout):
        x = self.x
        sech2 = 1.0 - self.t**2
        grad = 0.5 * (1 + self.t) + (0.5 * x * sech2 * 0.7978845608 * (1 + 3 * 0.044715 * x**2))
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

class SovereignBlock:
    def __init__(self, dim, expansion=4):
        self.ln = LayerNorm(dim)
        self.l1 = Linear(dim, dim * expansion)
        self.act = FastGELU()
        self.l2 = Linear(dim * expansion, dim)

    def forward(self, x):
        self.res = x
        h = self.ln.forward(x)
        h = self.l1.forward(h)
        h = self.act.forward(h)
        h = self.l2.forward(h)
        return h + self.res

    def backward(self, dout):
        dh = self.l2.backward(dout)
        dh = self.act.backward(dh)
        dh = self.l1.backward(dh)
        dh = self.ln.backward(dh)
        return dh + dout

    def get_params(self):
        return [self.l1.W, self.l1.b, self.l2.W, self.l2.b, self.ln.gamma, self.ln.beta]

    def get_grads(self):
        return [self.l1.dW, self.l1.db, self.l2.dW, self.l2.db, self.ln.dgamma, self.ln.dbeta]

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

class ConsensusProtocol:
    def __init__(self, threshold_loss=5.0, threshold_gnorm=10.0):
        self.threshold_loss = threshold_loss
        self.threshold_gnorm = threshold_gnorm
        self.history = []

    def validate(self, loss, gnorm):
        consensus = loss < self.threshold_loss and gnorm < self.threshold_gnorm
        self.history.append(consensus)
        return consensus

class SovereignArchitect:
    def __init__(self, in_d=784, h_d=256, out_d=10, depth=3):
        self.stem = Linear(in_d, h_d)
        self.blocks = [SovereignBlock(h_d) for _ in range(depth)]
        self.head_ln = LayerNorm(h_d)
        self.head = Linear(h_d, out_d)

        self.layers = [self.stem] + self.blocks + [self.head_ln, self.head]
        self.params = []
        for l in self.layers:
            if hasattr(l, 'get_params'):
                self.params.extend(l.get_params())
            else:
                self.params.extend([l.W, l.b] if hasattr(l, 'W') else [l.gamma, l.beta])

        self.optimizer = AdamW(self.params, lr=1e-3, wd=0.05)
        self.consensus = ConsensusProtocol()

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def backward(self, dout):
        for l in reversed(self.layers):
            dout = l.backward(dout)

        grads = []
        for l in self.layers:
            if hasattr(l, 'get_grads'):
                grads.extend(l.get_grads())
            else:
                grads.extend([l.dW, l.db] if hasattr(l, 'dW') else [l.dgamma, l.dbeta])
        return grads

    def evolve(self, grads, loss, lr_mult):
        gnorm = np.sqrt(sum(np.sum(g**2) for g in grads))
        if gnorm > 1.0:
            grads = [g * (1.0 / gnorm) for g in grads]

        if self.consensus.validate(loss, gnorm):
            self.optimizer.step(self.params, grads, lr_mult)
        else:
            self.optimizer.step(self.params, grads, lr_mult * 0.1)
        return gnorm

def run_recursive_evolution():
    N, D, K = 5000, 784, 10
    X = np.random.randn(N, D).astype(np.float32)
    W_target = np.random.randn(D, K).astype(np.float32)
    Y = np.argmax(np.dot(X, W_target) + np.random.randn(N, K) * 0.05, axis=1)

    model = SovereignArchitect(in_d=D, h_d=128, out_d=K, depth=2)
    batch_size = 128
    epochs = 50

    print("OMEGA-ASI SYSTEM ONLINE | ARCHITECTURE: MODULAR_SOVEREIGN")
    start_time = time.time()

    for epoch in range(epochs):
        indices = np.random.permutation(N)
        epoch_loss, epoch_acc, epoch_gnorm = 0, 0, 0
        lr_mult = 0.5 * (1 + np.cos(np.pi * epoch / epochs))

        for i in range(0, N, batch_size):
            idx = indices[i:i+batch_size]
            xb, yb = X[idx], Y[idx]
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
            gnorm = model.evolve(grads, loss, lr_mult)

            epoch_loss += loss * (m / N)
            epoch_acc += acc * (m / N)
            epoch_gnorm += gnorm * (m / N)

        if epoch % 5 == 0:
            status = "STABLE" if model.consensus.history[-1] else "DEGRADED"
            print(f"EPOCH:{epoch:03d} | LOSS:{epoch_loss:.4f} | ACC:{epoch_acc:.4f} | GNORM:{epoch_gnorm:.3f} | CONSENSUS:{status}")

    total_time = time.time() - start_time
    print(f"EVOLUTION_COMPLETE | TIME:{total_time:.2f}s | FINAL_ACC:{epoch_acc:.4f}")

if __name__ == "__main__":
    run_recursive_evolution()
