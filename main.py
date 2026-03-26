import numpy as np
import time
import os

class Parameter:
    def __init__(self, data, name=""):
        self.data = data.astype(np.float32)
        self.grad = np.zeros_like(data, dtype=np.float32)
        self.m = np.zeros_like(data, dtype=np.float32)
        self.v = np.zeros_like(data, dtype=np.float32)
        self.name = name

class Module:
    def __init__(self):
        self.params = []
        self.training = True

    def forward(self, x): raise NotImplementedError
    def backward(self, grad): raise NotImplementedError
    def get_params(self): return self.params

class Linear(Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        self.w = Parameter(np.random.uniform(-limit, limit, (in_dim, out_dim)), "w")
        self.b = Parameter(np.zeros((1, out_dim)), "b")
        self.params = [self.w, self.b]

    def forward(self, x):
        self.x = x
        return np.dot(x, self.w.data) + self.b.data

    def backward(self, grad):
        self.w.grad = np.dot(self.x.T, grad)
        self.b.grad = np.sum(grad, axis=0, keepdims=True)
        return np.dot(grad, self.w.data.T)

class LayerNorm(Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = Parameter(np.ones((1, dim)), "gamma")
        self.beta = Parameter(np.zeros((1, dim)), "beta")
        self.params = [self.gamma, self.beta]
        self.eps = eps

    def forward(self, x):
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.x_hat = (x - self.mean) / np.sqrt(self.var + self.eps)
        return self.gamma.data * self.x_hat + self.beta.data

    def backward(self, grad):
        n, d = grad.shape
        self.gamma.grad = np.sum(grad * self.x_hat, axis=0, keepdims=True)
        self.beta.grad = np.sum(grad, axis=0, keepdims=True)
        dx_hat = grad * self.gamma.data
        return (d * dx_hat - np.sum(dx_hat, axis=-1, keepdims=True) - self.x_hat * np.sum(dx_hat * self.x_hat, axis=-1, keepdims=True)) / (d * np.sqrt(self.var + self.eps))

class SiLU(Module):
    def forward(self, x):
        self.sig = 1.0 / (1.0 + np.exp(-x))
        self.x = x
        return x * self.sig

    def backward(self, grad):
        return grad * (self.sig * (1.0 + self.x * (1.0 - self.sig)))

class Dropout(Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0: return x
        self.mask = (np.random.rand(*x.shape) > self.p) / (1.0 - self.p)
        return x * self.mask

    def backward(self, grad):
        if not self.training or self.p == 0: return grad
        return grad * self.mask

class ResidualBlock(Module):
    def __init__(self, dim, dropout_p=0.1):
        super().__init__()
        self.ln = LayerNorm(dim)
        self.l1 = Linear(dim, dim * 4)
        self.act = SiLU()
        self.l2 = Linear(dim * 4, dim)
        self.drop = Dropout(dropout_p)
        self.params = self.ln.params + self.l1.params + self.l2.params

    def forward(self, x):
        self.res = x
        out = self.ln.forward(x)
        out = self.l1.forward(out)
        out = self.act.forward(out)
        out = self.l2.forward(out)
        return self.drop.forward(out) + self.res

    def backward(self, grad):
        dx = self.drop.backward(grad)
        dx = self.l2.backward(dx)
        dx = self.act.backward(dx)
        dx = self.l1.backward(dx)
        dx = self.ln.backward(dx)
        return dx + grad

class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=0.01):
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.wd = wd
        self.t = 0

    def step(self):
        self.t += 1
        b1, b2 = self.betas
        lr_t = self.lr * (np.sqrt(1.0 - b2**self.t) / (1.0 - b1**self.t))
        for p in self.params:
            if self.wd > 0: p.data -= self.lr * self.wd * p.data
            p.m = b1 * p.m + (1.0 - b1) * p.grad
            p.v = b2 * p.v + (1.0 - b2) * (p.grad**2)
            p.data -= lr_t * p.m / (np.sqrt(p.v) + self.eps)

class ConsensusSupervisor:
    def __init__(self, model):
        self.model = model
        self.loss_history = []
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")

    def audit(self, loss):
        self.loss_history.append(loss)
        if len(self.loss_history) < 5: return "WARMUP"
        
        window = self.loss_history[-5:]
        slope = np.polyfit(np.arange(5), window, 1)[0]
        
        # Groq Redundant Logic: High-speed throughput analysis
        groq_signal = "STABLE"
        if slope > 0.02: groq_signal = "REDUCE"
        elif slope < -0.01: groq_signal = "BOOST"
        
        # Gemini Redundant Logic: Deep context/loss landscape analysis
        gemini_signal = "STABLE"
        if loss > 2.5 and slope > 0: gemini_signal = "REDUCE"
        elif loss < 0.5 and slope > -0.001: gemini_signal = "BOOST"

        if groq_signal == "REDUCE" or gemini_signal == "REDUCE":
            self.model.optimizer.lr *= 0.7
            return "DEFLATION"
        if groq_signal == "BOOST" and gemini_signal == "BOOST":
            self.model.optimizer.lr *= 1.1
            return "EXPANSION"
        return "OPTIMAL"

class OMEGA_ASI:
    def __init__(self, in_d, hid_d, out_d, blocks=4):
        self.layers = [Linear(in_d, hid_d), LayerNorm(hid_d), SiLU()]
        for _ in range(blocks):
            self.layers.append(ResidualBlock(hid_d))
        self.layers.append(Linear(hid_d, out_d))
        
        self.params = []
        for l in self.layers: self.params.extend(l.get_params())
        self.optimizer = AdamW(self.params, lr=1e-3, wd=0.01)
        self.supervisor = ConsensusSupervisor(self)

    def forward(self, x, training=True):
        for l in self.layers:
            l.training = training
            x = l.forward(x)
        return x

    def backward(self, grad):
        for l in reversed(self.layers):
            grad = l.backward(grad)

    def train_step(self, x, y):
        logits = self.forward(x, training=True)
        # Softmax Cross Entropy
        shift = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(shift)
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        loss = -np.mean(np.sum(y * np.log(probs + 1e-12), axis=1))
        
        grad = (probs - y) / y.shape[0]
        self.backward(grad)
        self.optimizer.step()
        return loss

    def fit(self, x, y, epochs=50, batch_size=128):
        n = x.shape[0]
        for epoch in range(1, epochs + 1):
            idx = np.random.permutation(n)
            losses = []
            t0 = time.time()
            for i in range(0, n, batch_size):
                xb, yb = x[idx[i:i+batch_size]], y[idx[i:i+batch_size]]
                losses.append(self.train_step(xb, yb))
            
            avg_loss = np.mean(losses)
            status = self.supervisor.audit(avg_loss)
            dt = time.time() - t0
            
            if epoch % 5 == 0 or epoch == 1:
                acc = self.evaluate(x[:1000], y[:1000])
                print(f"EP {epoch:03d} | LOSS {avg_loss:.4f} | ACC {acc:.4f} | {dt:.2f}s | {status}")

    def evaluate(self, x, y):
        logits = self.forward(x, training=False)
        return np.mean(np.argmax(logits, axis=1) == np.argmax(y, axis=1))

def generate_synthetic_data(n=10000, d=784, c=10):
    x = np.random.randn(n, d).astype(np.float32)
    w = np.random.randn(d, c).astype(np.float32)
    y_idx = np.argmax(np.dot(x, w) + 0.1 * np.random.randn(n, c), axis=1)
    return x, np.eye(c)[y_idx].astype(np.float32)

if __name__ == "__main__":
    X, Y = generate_synthetic_data()
    model = OMEGA_ASI(784, 256, 10, blocks=3)
    model.fit(X, Y, epochs=50, batch_size=256)
