import numpy as np
import time
import os
import sys

class Tensor:
    def __init__(self, data, name=""):
        self.data = data
        self.grad = np.zeros_like(data)
        self.name = name

class GELU:
    def forward(self, x):
        self.x = x
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    def backward(self, grad):
        sech = 1 / np.cosh(np.sqrt(2 / np.pi) * (self.x + 0.044715 * np.power(self.x, 3)))
        deriv = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (self.x + 0.044715 * np.power(self.x, 3)))) + \
                (0.5 * self.x * (sech**2) * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * self.x**2))
        return grad * deriv

class BatchNorm:
    def __init__(self, dim, momentum=0.99, eps=1e-5):
        self.gamma = Tensor(np.ones((1, dim)), "gamma")
        self.beta = Tensor(np.zeros((1, dim)), "beta")
        self.running_mean = np.zeros((1, dim))
        self.running_var = np.ones((1, dim))
        self.momentum = momentum
        self.eps = eps
        self.training = True

    def forward(self, x):
        if not self.training:
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            return self.gamma.data * x_hat + self.beta.data
        
        mu = np.mean(x, axis=0, keepdims=True)
        var = np.var(x, axis=0, keepdims=True)
        self.x_hat = (x - mu) / np.sqrt(var + self.eps)
        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
        self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        self.mu, self.var = mu, var
        return self.gamma.data * self.x_hat + self.beta.data

    def backward(self, grad):
        N = grad.shape[0]
        self.gamma.grad = np.sum(grad * self.x_hat, axis=0, keepdims=True)
        self.beta.grad = np.sum(grad, axis=0, keepdims=True)
        dx_hat = grad * self.gamma.data
        dvar = np.sum(dx_hat * (self.x_hat * -0.5 / (self.var + self.eps)), axis=0, keepdims=True)
        dmu = np.sum(dx_hat * -1 / np.sqrt(self.var + self.eps), axis=0, keepdims=True)
        return dx_hat / np.sqrt(self.var + self.eps) + dvar * 2 * self.x_hat / N + dmu / N

class Linear:
    def __init__(self, in_dim, out_dim):
        scale = np.sqrt(2.0 / in_dim)
        self.w = Tensor(np.random.randn(in_dim, out_dim) * scale, "weight")
        self.b = Tensor(np.zeros((1, out_dim)), "bias")

    def forward(self, x):
        self.x = x
        return np.dot(x, self.w.data) + self.b.data

    def backward(self, grad):
        self.w.grad = np.dot(self.x.T, grad)
        self.b.grad = np.sum(grad, axis=0, keepdims=True)
        return np.dot(grad, self.w.data.T)

class ResidualBlock:
    def __init__(self, dim, dropout_p=0.1):
        self.ln1 = BatchNorm(dim)
        self.l1 = Linear(dim, dim * 2)
        self.act = GELU()
        self.l2 = Linear(dim * 2, dim)
        self.drop = Dropout(dropout_p)

    def forward(self, x):
        self.residual = x
        out = self.ln1.forward(x)
        out = self.l1.forward(out)
        out = self.act.forward(out)
        out = self.l2.forward(out)
        out = self.drop.forward(out)
        return out + self.residual

    def backward(self, grad):
        dr = grad
        dg = self.drop.backward(grad)
        dg = self.l2.backward(dg)
        dg = self.act.backward(dg)
        dg = self.l1.backward(dg)
        dg = self.ln1.backward(dg)
        return dg + dr

class Dropout:
    def __init__(self, p=0.1):
        self.p = p
        self.training = True

    def forward(self, x):
        if not self.training or self.p == 0: return x
        self.mask = (np.random.rand(*x.shape) > self.p) / (1.0 - self.p)
        return x * self.mask

    def backward(self, grad):
        if not self.training or self.p == 0: return grad
        return grad * self.mask

class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=0.01):
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.wd = wd
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        lr_t = self.lr * (np.sqrt(1.0 - self.betas[1]**self.t) / (1.0 - self.betas[0]**self.t))
        for i, p in enumerate(self.params):
            p.data -= self.lr * self.wd * p.data
            self.m[i] = self.betas[0] * self.m[i] + (1.0 - self.betas[0]) * p.grad
            self.v[i] = self.betas[1] * self.v[i] + (1.0 - self.betas[1]) * (p.grad**2)
            p.data -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + self.eps)

class ConsensusSupervisor:
    def __init__(self):
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.telemetry = []

    def validate_evolution(self, loss, optimizer):
        self.telemetry.append(loss)
        if len(self.telemetry) < 5: return "WARMUP"
        
        recent_avg = np.mean(self.telemetry[-5:])
        delta = self.telemetry[-1] - recent_avg
        
        # Simulated Redundant Logic Gates
        groq_signal = self._groq_inference(delta)
        gemini_signal = self._gemini_inference(delta)
        
        if groq_signal == "PANIC" or gemini_signal == "PANIC":
            optimizer.lr *= 0.5
            return "CRITICAL_DEFLATION"
        
        if groq_signal == "STAGNANT" and gemini_signal == "STAGNANT":
            optimizer.lr *= 1.1
            return "MOMENTUM_INJECTION"
            
        return "STABLE"

    def _groq_inference(self, delta):
        if not self.groq_key: return "STABLE" if delta < 0 else "PANIC"
        return "STABLE" if delta < 0.01 else "PANIC"

    def _gemini_inference(self, delta):
        if not self.gemini_key: return "STABLE" if abs(delta) > 1e-5 else "STAGNANT"
        return "STABLE" if abs(delta) > 1e-6 else "STAGNANT"

class OMEGA_ASI:
    def __init__(self, in_d, hid_d, out_d, blocks=2):
        self.layers = [Linear(in_d, hid_d), BatchNorm(hid_d), GELU()]
        for _ in range(blocks):
            self.layers.append(ResidualBlock(hid_d))
        self.layers.append(Linear(hid_d, out_d))
        
        self.params = []
        self._collect_params(self.layers)
        self.optimizer = AdamW(self.params, lr=1e-3, wd=0.05)
        self.supervisor = ConsensusSupervisor()

    def _collect_params(self, layers):
        for l in layers:
            if hasattr(l, 'w'): self.params.extend([l.w, l.b])
            if hasattr(l, 'gamma'): self.params.extend([l.gamma, l.beta])
            if hasattr(l, 'l1'): self._collect_params([l.ln1, l.l1, l.l2])

    def forward(self, x, training=True):
        for l in self.layers:
            if hasattr(l, 'training'): l.training = training
            x = l.forward(x)
        return x

    def backward(self, grad):
        for l in reversed(self.layers):
            grad = l.backward(grad)

    def compute_loss(self, logits, y):
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        self.probs = probs
        return -np.mean(np.sum(y * np.log(probs + 1e-12), axis=1))

    def train_step(self, x, y):
        logits = self.forward(x, training=True)
        loss = self.compute_loss(logits, y)
        grad = (self.probs - y) / y.shape[0]
        self.backward(grad)
        self.optimizer.step()
        return loss

    def fit(self, x, y, epochs=50, batch_size=128):
        n = x.shape[0]
        for epoch in range(1, epochs + 1):
            idx = np.random.permutation(n)
            epoch_losses = []
            t0 = time.time()
            for i in range(0, n, batch_size):
                xb, yb = x[idx[i:i+batch_size]], y[idx[i:i+batch_size]]
                epoch_losses.append(self.train_step(xb, yb))
            
            avg_loss = np.mean(epoch_losses)
            status = self.supervisor.validate_evolution(avg_loss, self.optimizer)
            
            if epoch % 5 == 0 or epoch == 1:
                acc = self.evaluate(x[:1000], y[:1000])
                print(f"| EPOCH {epoch:03d} | LOSS: {avg_loss:.4f} | ACC: {acc:.4f} | {time.time()-t0:.2f}s | {status} |")

    def evaluate(self, x, y):
        logits = self.forward(x, training=False)
        return np.mean(np.argmax(logits, axis=1) == np.argmax(y, axis=1))

def generate_high_dim_data(n=5000, d=784, c=10):
    x = np.random.standard_normal((n, d)).astype(np.float32)
    w = np.random.randn(d, c)
    y_idx = np.argmax(np.dot(x, w) + np.random.normal(0, 0.5, (n, c)), axis=1)
    y = np.eye(c)[y_idx]
    return x, y

if __name__ == "__main__":
    print("--- OMEGA-ASI: SOVEREIGN ARCHITECT INITIALIZED ---")
    X, Y = generate_high_dim_data()
    model = OMEGA_ASI(784, 512, 10, blocks=3)
    model.fit(X, Y, epochs=50, batch_size=256)
    print("--- RECURSIVE EVOLUTION CYCLE COMPLETE ---")
