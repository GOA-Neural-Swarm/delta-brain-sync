
import numpy as np
import time

class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=0.01):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.wd = wd
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for i in range(len(params)):
            params[i] -= self.lr * self.wd * params[i]
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i]**2)
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

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
        dx_hat = dout * self.gamma
        dvar = np.sum(dx_hat * (self.x - self.mu) * -0.5 * self.std_inv**3, axis=-1, keepdims=True)
        dmu = np.sum(dx_hat * -self.std_inv, axis=-1, keepdims=True) + dvar * np.mean(-2.0 * (self.x - self.mu), axis=-1, keepdims=True)
        dx = dx_hat * self.std_inv + dvar * 2.0 * (self.x - self.mu) / D + dmu / D
        self.dgamma = np.sum(dout * self.x_hat, axis=0, keepdims=True)
        self.dbeta = np.sum(dout, axis=0, keepdims=True)
        return dx

    def get_params(self): return [self.gamma, self.beta]
    def set_params(self, p): self.gamma, self.beta = p
    def get_grads(self): return [self.dgamma, self.dbeta]

class Swish:
    def forward(self, x):
        self.x = x
        self.sig = 1.0 / (1.0 + np.exp(-x))
        return x * self.sig

    def backward(self, dout):
        return dout * (self.sig + self.x * self.sig * (1.0 - self.sig))

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
    def set_params(self, p): self.W, self.b = p
    def get_grads(self): return [self.dW, self.db]

class ResidualBlock:
    def __init__(self, dim):
        self.ln = LayerNorm(dim)
        self.l1 = Linear(dim, dim * 2)
        self.act = Swish()
        self.l2 = Linear(dim * 2, dim)

    def forward(self, x):
        self.res = x
        h = self.ln.forward(x)
        h = self.l1.forward(h)
        h = self.act.forward(h)
        h = self.l2.forward(h)
        return h + x

    def backward(self, dout):
        dh = self.l2.backward(dout)
        dh = self.act.backward(dh)
        dh = self.l1.backward(dh)
        dh = self.ln.backward(dh)
        return dh + dout

    def get_layers(self): return [self.ln, self.l1, self.l2]

class ExpertModule:
    def __init__(self, in_d, h_d, out_d, depth=2):
        self.proj_in = Linear(in_d, h_d)
        self.blocks = [ResidualBlock(h_d) for _ in range(depth)]
        self.proj_out = Linear(h_d, out_d)

    def forward(self, x):
        x = self.proj_in.forward(x)
        for b in self.blocks: x = b.forward(x)
        return self.proj_out.forward(x)

    def backward(self, dout):
        dout = self.proj_out.backward(dout)
        for b in reversed(self.blocks): dout = b.backward(dout)
        return self.proj_in.backward(dout)

    def get_layers(self):
        layers = [self.proj_in]
        for b in self.blocks: layers.extend(b.get_layers())
        layers.append(self.proj_out)
        return layers

class SovereignArchitect:
    def __init__(self, in_d=784, h_d=256, out_d=10):
        self.expert = ExpertModule(in_d, h_d, out_d)
        self.gate_w = np.random.randn(1, out_d).astype(np.float32) * 0.01
        self.layers = self.expert.get_layers()
        params = [l.get_params() for l in self.layers]
        self.flat_params = [p for sub in params for p in sub]
        self.flat_params.append(self.gate_w)
        self.optimizer = AdamW(self.flat_params)

    def _softmax(self, x):
        ex = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return ex / np.sum(ex, axis=-1, keepdims=True)

    def forward(self, x):
        self.out = self.expert.forward(x)
        self.gate_logits = self._softmax(self.gate_w)
        return self.gate_logits * self.out

    def backward(self, dout):
        dgate = np.sum(dout * self.out, axis=0, keepdims=True)
        self.expert.backward(dout * self.gate_logits)
        grads = [l.get_grads() for l in self.layers]
        flat_grads = [g for sub in grads for g in sub]
        flat_grads.append(dgate)
        self.optimizer.step(self.flat_params, flat_grads)

class EvolutionOrchestrator:
    def __init__(self):
        self.model = SovereignArchitect()
        self.data_x = np.random.randn(10000, 784).astype(np.float32)
        self.data_y = np.random.randint(0, 10, 10000)
        self.batch_size = 128

    def train(self, steps=2000):
        print("PHASE: RECURSIVE_EVOLUTION_INITIATED")
        start_time = time.time()
        for s in range(steps):
            idx = np.random.randint(0, 10000, self.batch_size)
            x, y = self.data_x[idx], self.data_y[idx]

            logits = self.model.forward(x)
            probs = self.model._softmax(logits)
            loss = -np.mean(np.log(probs[range(self.batch_size), y] + 1e-10))

            grad = probs.copy()
            grad[range(self.batch_size), y] -= 1
            grad /= self.batch_size

            self.model.backward(grad)

            if s % 100 == 0:
                acc = np.mean(np.argmax(probs, axis=1) == y)
                elapsed = time.time() - start_time
                print(f"STEP:{s:04d} | LOSS:{loss:.4f} | ACC:{acc:.4f} | TIME:{elapsed:.2f}s")
                if s % 500 == 0: self.model.optimizer.lr *= 0.8

if __name__ == "__main__":
    orch = EvolutionOrchestrator()
    orch.train()
