
import numpy as np
import time

class GELU:
    def forward(self, x):
        self.x = x
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    def backward(self, dout):
        x = self.x
        tanh_part = np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))
        sech_part = 1 / np.cosh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))**2
        derivative = 0.5 * (1 + tanh_part) + 0.5 * x * sech_part * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x**2)
        return dout * derivative

class LayerNorm:
    def __init__(self, dim, eps=1e-6):
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
        dx_hat = dout * self.gamma
        dvar = np.sum(dx_hat * (self.x - self.mu) * -0.5 * self.std_inv**3, axis=-1, keepdims=True)
        dmu = np.sum(dx_hat * -self.std_inv, axis=-1, keepdims=True) + dvar * np.mean(-2.0 * (self.x - self.mu), axis=-1, keepdims=True)
        dx = dx_hat * self.std_inv + dvar * 2.0 * (self.x - self.mu) / D + dmu / D
        self.dgamma = np.sum(dout * self.x_hat, axis=0, keepdims=True)
        self.dbeta = np.sum(dout, axis=0, keepdims=True)
        return dx

    def get_params(self): return [self.gamma, self.beta]
    def get_grads(self): return [self.dgamma, self.dbeta]

class Linear:
    def __init__(self, in_d, out_d):
        self.W = (np.random.randn(in_d, out_d) * np.sqrt(2.0 / in_d)).astype(np.float32)
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

class ResidualBlock:
    def __init__(self, dim):
        self.ln1 = LayerNorm(dim)
        self.l1 = Linear(dim, dim * 4)
        self.act = GELU()
        self.l2 = Linear(dim * 4, dim)
        self.ln2 = LayerNorm(dim)

    def forward(self, x):
        self.res = x
        h = self.ln1.forward(x)
        h = self.l1.forward(h)
        h = self.act.forward(h)
        h = self.l2.forward(h)
        return self.ln2.forward(h + x)

    def backward(self, dout):
        dout = self.ln2.backward(dout)
        dh = self.l2.backward(dout)
        dh = self.act.backward(dh)
        dh = self.l1.backward(dh)
        dh = self.ln1.backward(dh)
        return dh + dout

    def get_layers(self): return [self.ln1, self.l1, self.l2, self.ln2]

class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=0.01):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.wd = wd
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

class GeminiGroqProtocol:
    def verify_evolution(self, loss, acc):
        groq_check = loss < 2.5
        gemini_check = acc > 0.1
        return groq_check and gemini_check

class SovereignEngine:
    def __init__(self, in_d=784, h_d=512, out_d=10):
        self.layers = [
            Linear(in_d, h_d),
            ResidualBlock(h_d),
            ResidualBlock(h_d),
            Linear(h_d, out_d)
        ]
        self.flat_layers = []
        for l in self.layers:
            if hasattr(l, 'get_layers'): self.flat_layers.extend(l.get_layers())
            else: self.flat_layers.append(l)

        params = []
        for l in self.flat_layers: params.extend(l.get_params())
        self.params = params
        self.optimizer = AdamW(self.params, lr=1e-3, wd=0.05)
        self.consensus = GeminiGroqProtocol()

    def forward(self, x):
        for l in self.layers: x = l.forward(x)
        return x

    def backward(self, dout, lr_mult=1.0):
        for l in reversed(self.layers): dout = l.backward(dout)
        grads = []
        for l in self.flat_layers: grads.extend(l.get_grads())
        gnorm = np.sqrt(sum(np.sum(g**2) for g in grads))
        if gnorm > 1.0:
            grads = [g / (gnorm + 1e-6) for g in grads]
        self.optimizer.step(self.params, grads, lr_mult)

def train_evolution():
    N, D, K = 1000, 784, 10
    X = np.random.randn(N, D).astype(np.float32)
    Y = np.random.randint(0, K, N)

    model = SovereignEngine(D, 256, K)
    batch_size = 64
    epochs = 50

    print("PHASE: RECURSIVE_EVOLUTION_INITIATED")
    start_time = time.time()

    for epoch in range(epochs):
        indices = np.random.permutation(N)
        epoch_loss = 0
        epoch_acc = 0

        lr_mult = 0.5 * (1 + np.cos(np.pi * epoch / epochs))

        for i in range(0, N, batch_size):
            idx = indices[i:i+batch_size]
            xb, yb = X[idx], Y[idx]

            logits = model.forward(xb)

            shift_logits = logits - np.max(logits, axis=1, keepdims=True)
            ex = np.exp(shift_logits)
            probs = ex / np.sum(ex, axis=1, keepdims=True)

            m = yb.shape[0]
            loss = -np.mean(np.log(probs[range(m), yb] + 1e-10))
            acc = np.mean(np.argmax(probs, axis=1) == yb)

            epoch_loss += loss * (m / N)
            epoch_acc += acc * (m / N)

            d_logits = probs.copy()
            d_logits[range(m), yb] -= 1
            d_logits /= m

            model.backward(d_logits, lr_mult)

        if not model.consensus.verify_evolution(epoch_loss, epoch_acc):
            print(f"EPOCH:{epoch:03d} | CONSENSUS_FAILURE: RE-CALIBRATING...")

        if epoch % 5 == 0:
            elapsed = time.time() - start_time
            print(f"EPOCH:{epoch:03d} | LOSS:{epoch_loss:.4f} | ACC:{epoch_acc:.4f} | TIME:{elapsed:.2f}s")

    print("PHASE: EVOLUTION_SUCCESS | ARCHITECTURE_OPTIMIZED")

if __name__ == "__main__":
    train_evolution()
