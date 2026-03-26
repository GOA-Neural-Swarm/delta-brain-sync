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
        # Kaiming Initialization
        scale = np.sqrt(2.0 / in_dim)
        self.w = Parameter(np.random.randn(in_dim, out_dim) * scale, "w")
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

class GELU(Module):
    def forward(self, x):
        self.x = x
        # Fast approximation: 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3)))
        self.tanh_in = np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))
        self.tanh_out = np.tanh(self.tanh_in)
        return 0.5 * x * (1 + self.tanh_out)

    def backward(self, grad):
        # Derivative approximation
        sech2 = 1 - self.tanh_out**2
        d_tanh = sech2 * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * self.x**2)
        return grad * (0.5 * (1 + self.tanh_out) + 0.5 * self.x * d_tanh)

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
        # Pre-Norm Architecture
        self.ln = LayerNorm(dim)
        self.l1 = Linear(dim, dim * 4)
        self.act = GELU()
        self.l2 = Linear(dim * 4, dim)
        self.drop = Dropout(dropout_p)
        self.params = self.ln.params + self.l1.params + self.l2.params

    def forward(self, x):
        self.res = x
        out = self.ln.forward(x)
        out = self.l1.forward(out)
        out = self.act.forward(out)
        out = self.l2.forward(out)
        out = self.drop.forward(out)
        return out + self.res

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
            if self.wd > 0:
                p.data -= self.lr * self.wd * p.data
            p.m = b1 * p.m + (1.0 - b1) * p.grad
            p.v = b2 * p.v + (1.0 - b2) * (p.grad**2)
            p.data -= lr_t * p.m / (np.sqrt(p.v) + self.eps)

class ConsensusSupervisor:
    def __init__(self, model):
        self.model = model
        self.history = []
        self.grad_norms = []

    def audit(self, loss, params):
        self.history.append(loss)
        
        # Calculate Gradient Norm for Groq Logic
        total_norm = np.sqrt(sum(np.sum(p.grad**2) for p in params))
        self.grad_norms.append(total_norm)
        
        if len(self.history) < 10: return "WARMUP"

        # Groq Redundant Logic: Throughput & Stability (Gradient Variance)
        # High grad norm suggests instability; low suggests stagnation
        groq_signal = "STABLE"
        if total_norm > 5.0: groq_signal = "REDUCE"
        elif total_norm < 0.1: groq_signal = "BOOST"

        # Gemini Redundant Logic: Contextual Convergence (Loss Landscape)
        # Analyzes the second derivative (curvature) of the loss history
        gemini_signal = "STABLE"
        recent_loss = self.history[-10:]
        slope = np.polyfit(np.arange(10), recent_loss, 1)[0]
        curvature = np.gradient(np.gradient(recent_loss)).mean()

        if slope > 0: gemini_signal = "REDUCE" # Diverging
        elif abs(slope) < 1e-4 and curvature > 0: gemini_signal = "BOOST" # Local Minima

        # Consensus Decision
        if groq_signal == "REDUCE" or gemini_signal == "REDUCE":
            self.model.optimizer.lr *= 0.5
            return f"DEFLATION (G:{groq_signal} M:{gemini_signal})"
        if groq_signal == "BOOST" and gemini_signal == "BOOST":
            self.model.optimizer.lr = min(self.model.optimizer.lr * 1.2, 1e-2)
            return f"EXPANSION (G:{groq_signal} M:{gemini_signal})"
        
        return "OPTIMAL"

class OMEGA_ASI:
    def __init__(self, in_d, hid_d, out_d, blocks=6):
        self.layers = [Linear(in_d, hid_d), LayerNorm(hid_d), GELU()]
        for _ in range(blocks):
            self.layers.append(ResidualBlock(hid_d))
        self.layers.append(Linear(hid_d, out_d))
        
        self.params = []
        for l in self.layers: self.params.extend(l.get_params())
        self.optimizer = AdamW(self.params, lr=2e-3, wd=0.05)
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
        # Cross Entropy with LogSumExp for stability
        max_l = np.max(logits, axis=1, keepdims=True)
        log_sum_exp = max_l + np.log(np.sum(np.exp(logits - max_l), axis=1, keepdims=True))
        probs = np.exp(logits - log_sum_exp)
        loss = -np.mean(np.sum(y * (logits - log_sum_exp), axis=1))
        
        grad = (probs - y) / y.shape[0]
        self.backward(grad)
        self.optimizer.step()
        return loss

    def fit(self, x, y, epochs=100, batch_size=256):
        n = x.shape[0]
        best_acc = 0
        for epoch in range(1, epochs + 1):
            idx = np.random.permutation(n)
            losses = []
            t0 = time.time()
            for i in range(0, n, batch_size):
                xb, yb = x[idx[i:i+batch_size]], y[idx[i:i+batch_size]]
                losses.append(self.train_step(xb, yb))
            
            avg_loss = np.mean(losses)
            status = self.supervisor.audit(avg_loss, self.params)
            dt = time.time() - t0
            
            if epoch % 2 == 0 or epoch == 1:
                acc = self.evaluate(x[:2000], y[:2000])
                best_acc = max(best_acc, acc)
                print(f"EPOCH {epoch:03d} | LOSS: {avg_loss:.5f} | ACC: {acc:.4f} | LR: {self.optimizer.lr:.6f} | {dt:.2f}s | {status}")
                if acc > 0.99: break

    def evaluate(self, x, y):
        logits = self.forward(x, training=False)
        return np.mean(np.argmax(logits, axis=1) == np.argmax(y, axis=1))

def generate_complex_synthetic(n=15000, d=784, c=10):
    # Generate non-linear clusters
    x = np.random.randn(n, d).astype(np.float32)
    # Apply non-linear transformation to create labels
    w1 = np.random.randn(d, 512)
    w2 = np.random.randn(512, c)
    z = np.dot(np.maximum(0, np.dot(x, w1)), w2)
    y_idx = np.argmax(z + 0.05 * np.random.randn(n, c), axis=1)
    return x, np.eye(c)[y_idx].astype(np.float32)

if __name__ == "__main__":
    print("INITIALIZING OMEGA-ASI ARCHITECTURE...")
    X, Y = generate_complex_synthetic()
    # Normalize inputs
    X = (X - np.mean(X)) / (np.std(X) + 1e-8)
    
    model = OMEGA_ASI(in_d=784, hid_d=128, out_d=10, blocks=4)
    model.fit(X, Y, epochs=100, batch_size=512)
