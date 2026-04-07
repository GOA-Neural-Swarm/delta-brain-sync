
import numpy as np
import time

class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=0.01):
        self.lr, self.beta1, self.beta2, self.eps, self.wd = lr, betas[0], betas[1], eps, wd
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)
        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i]**2)
            params[i] -= lr_t * (self.m[i] / (np.sqrt(self.v[i]) + self.eps) + self.wd * params[i])

class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones((1, dim), dtype=np.float32)
        self.beta = np.zeros((1, dim), dtype=np.float32)
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
        self.dgamma = np.sum(dout * self.x_hat, axis=0, keepdims=True)
        self.dbeta = np.sum(dout, axis=0, keepdims=True)
        dx_hat = dout * self.gamma
        dx = (1. / D) * self.std_inv * (D * dx_hat - np.sum(dx_hat, axis=-1, keepdims=True) - self.x_hat * np.sum(dx_hat * self.x_hat, axis=-1, keepdims=True))
        return dx

    def get_params(self): return [self.gamma, self.beta]
    def get_grads(self): return [self.dgamma, self.dbeta]

class GeLU:
    def forward(self, x):
        self.x = x
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    def backward(self, dout):
        x = self.x
        tanh_part = np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))
        pdf = (1 / np.cosh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))**2
        derivative = 0.5 * (1 + tanh_part) + 0.5 * x * pdf * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x**2)
        return dout * derivative

class Linear:
    def __init__(self, in_d, out_d):
        limit = np.sqrt(6 / (in_d + out_d))
        self.W = np.random.uniform(-limit, limit, (in_d, out_d)).astype(np.float32)
        self.b = np.zeros((1, out_d), dtype=np.float32)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0, keepdims=True)
        return dout @ self.W.T

    def get_params(self): return [self.W, self.b]
    def get_grads(self): return [self.dW, self.db]

class GeminiGroqBlock:
    def __init__(self, dim):
        self.ln = LayerNorm(dim)
        self.l_gate = Linear(dim, dim * 2)
        self.l_proj = Linear(dim * 2, dim)
        self.act = GeLU()

    def forward(self, x):
        self.res = x
        h = self.ln.forward(x)
        h = self.l_gate.forward(h)

        self.h_a, self.h_b = np.split(h, 2, axis=-1)
        gated = self.act.forward(self.h_a) * self.h_b

        self.h_gated = np.concatenate([gated, gated], axis=-1)
        h = self.l_proj.forward(self.h_gated)
        return h + x

    def backward(self, dout):
        dout_proj = self.l_proj.backward(dout)

        dg_a_full, dg_b_full = np.split(dout_proj, 2, axis=-1)
        dg = dg_a_full + dg_b_full

        dh_a = self.act.backward(dg * self.h_b)
        dh_b = dg * self.act.forward(self.h_a)

        dh_gate = np.concatenate([dh_a, dh_b], axis=-1)
        dh = self.l_gate.backward(dh_gate)
        return self.ln.backward(dh) + dout

    def get_layers(self): return [self.ln, self.l_gate, self.l_proj]

class SovereignEngine:
    def __init__(self, in_d=784, h_d=256, out_d=10, num_blocks=4):
        self.stem = Linear(in_d, h_d)
        self.blocks = [GeminiGroqBlock(h_d) for _ in range(num_blocks)]
        self.head = Linear(h_d, out_d)

        self.all_layers = [self.stem]
        for b in self.blocks: self.all_layers.extend(b.get_layers())
        self.all_layers.append(self.head)

        self.params = []
        for l in self.all_layers: self.params.extend(l.get_params())
        self.optimizer = AdamW(self.params, lr=2e-3, wd=0.01)

    def forward(self, x):
        x = self.stem.forward(x)
        for b in self.blocks: x = b.forward(x)
        return self.head.forward(x)

    def backward(self, dout):
        dout = self.head.backward(dout)
        for b in reversed(self.blocks): dout = b.backward(dout)
        self.stem.backward(dout)

        grads = []
        for l in self.all_layers: grads.extend(l.get_grads())
        self.optimizer.step(self.params, grads)

def train_evolution():
    np.random.seed(42)
    N, D, C = 1024, 784, 10
    X = np.random.randn(N, D).astype(np.float32)
    Y = np.random.randint(0, C, N)

    batch_size = 128
    model = SovereignEngine(in_d=D, h_d=128, out_d=C, num_blocks=3)

    print("--- OMEGA-ASI: RECURSIVE EVOLUTION INITIATED ---")
    start_time = time.time()

    for epoch in range(1, 151):
        idx = np.random.permutation(N)
        epoch_loss = 0
        epoch_acc = 0

        for i in range(0, N, batch_size):
            batch_idx = idx[i:i+batch_size]
            xb, yb = X[batch_idx], Y[batch_idx]

            logits = model.forward(xb)

            exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exps / np.sum(exps, axis=1, keepdims=True)

            loss = -np.mean(np.log(probs[range(len(yb)), yb] + 1e-12))
            acc = np.mean(np.argmax(probs, axis=1) == yb)

            d_logits = probs.copy()
            d_logits[range(len(yb)), yb] -= 1
            d_logits /= len(yb)
            model.backward(d_logits)

            epoch_loss += loss * (len(yb) / N)
            epoch_acc += acc * (len(yb) / N)

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"STEP:{epoch:03d} | LOSS:{epoch_loss:.5f} | ACC:{epoch_acc:.4f} | T:{elapsed:.2f}s")

    print(f"--- EVOLUTION COMPLETE | FINAL ACC: {epoch_acc:.4f} ---")

if __name__ == "__main__":
    train_evolution()
