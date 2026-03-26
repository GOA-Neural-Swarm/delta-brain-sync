import numpy as np
import time
import sys

class Parameter:
    def __init__(self, data, name=""):
        self.data = data.astype(np.float32)
        self.grad = np.zeros_like(data)
        self.m = np.zeros_like(data)
        self.v = np.zeros_like(data)
        self.name = name

class Module:
    def __init__(self):
        self.params = []
        self.training = True
    def forward(self, x): raise NotImplementedError
    def backward(self, grad): raise NotImplementedError
    def get_params(self):
        params = []
        for attr in self.__dict__.values():
            if isinstance(attr, Parameter): params.append(attr)
            elif isinstance(attr, Module): params.extend(attr.get_params())
            elif isinstance(attr, list):
                for item in attr:
                    if isinstance(item, Module): params.extend(item.get_params())
        return params

class Linear(Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        scale = np.sqrt(2.0 / in_dim)
        self.w = Parameter(np.random.randn(in_dim, out_dim) * scale, "w")
        self.b = Parameter(np.zeros((1, out_dim)), "b")
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
        self.eps = eps
    def forward(self, x):
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.std = np.sqrt(self.var + self.eps)
        self.x_hat = (x - self.mean) / self.std
        return self.gamma.data * self.x_hat + self.beta.data
    def backward(self, grad):
        n, d = grad.shape
        self.gamma.grad = np.sum(grad * self.x_hat, axis=0, keepdims=True)
        self.beta.grad = np.sum(grad, axis=0, keepdims=True)
        dx_hat = grad * self.gamma.data
        return (d * dx_hat - np.sum(dx_hat, axis=-1, keepdims=True) - self.x_hat * np.sum(dx_hat * self.x_hat, axis=-1, keepdims=True)) / (d * self.std)

class GELU(Module):
    def forward(self, x):
        self.x = x
        self.tanh_in = 0.7978845608 * (x + 0.044715 * x**3)
        self.tanh_out = np.tanh(self.tanh_in)
        return 0.5 * x * (1 + self.tanh_out)
    def backward(self, grad):
        sech2 = 1 - self.tanh_out**2
        d_tanh = sech2 * 0.7978845608 * (1 + 3 * 0.044715 * self.x**2)
        return grad * (0.5 * (1 + self.tanh_out) + 0.5 * self.x * d_tanh)

class Dropout(Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p
        self.mask = None
    def forward(self, x):
        if not self.training or self.p == 0: return x
        self.mask = (np.random.rand(*x.shape) > self.p) / (1.0 - self.p)
        return x * self.mask
    def backward(self, grad):
        if not self.training or self.p == 0: return grad
        return grad * self.mask

class PreLNResidualBlock(Module):
    def __init__(self, dim, dropout_p=0.1):
        super().__init__()
        self.ln = LayerNorm(dim)
        self.l1 = Linear(dim, dim * 4)
        self.act = GELU()
        self.l2 = Linear(dim * 4, dim)
        self.drop = Dropout(dropout_p)
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
        self.grad_norms = []
    def audit(self, loss):
        self.loss_history.append(loss)
        gnorm = np.sqrt(sum(np.sum(p.grad**2) for p in self.model.params))
        self.grad_norms.append(gnorm)
        
        if len(self.loss_history) < 10: return "WARMUP"
        
        # Groq Logic: Gradient Signal-to-Noise Ratio & Magnitude Stability
        groq_signal = "STABLE"
        recent_gnorms = self.grad_norms[-10:]
        gnorm_std = np.std(recent_gnorms)
        gnorm_mean = np.mean(recent_gnorms)
        if gnorm_std / (gnorm_mean + 1e-8) > 0.5: groq_signal = "REDUCE"
        elif gnorm_mean < 1e-4: groq_signal = "BOOST"
        
        # Gemini Logic: Loss Curvature & Convergence Velocity
        gemini_signal = "STABLE"
        recent_loss = self.loss_history[-10:]
        slope = np.polyfit(np.arange(10), recent_loss, 1)[0]
        if slope > 0: gemini_signal = "REDUCE"
        elif abs(slope) < 1e-6: gemini_signal = "BOOST"
        
        if groq_signal == "REDUCE" or gemini_signal == "REDUCE":
            self.model.optimizer.lr *= 0.8
            return f"DEFLATE({groq_signal[0]}{gemini_signal[0]})"
        if groq_signal == "BOOST" and gemini_signal == "BOOST":
            self.model.optimizer.lr = min(self.model.optimizer.lr * 1.05, 1e-2)
            return f"EXPAND({groq_signal[0]}{gemini_signal[0]})"
        return "OPTIMAL"

class OMEGA_ASI:
    def __init__(self, in_d, hid_d, out_d, blocks=6):
        self.stem = Linear(in_d, hid_d)
        self.blocks = [PreLNResidualBlock(hid_d) for _ in range(blocks)]
        self.head_ln = LayerNorm(hid_d)
        self.head = Linear(hid_d, out_d)
        self.params = self.get_params()
        self.optimizer = AdamW(self.params, lr=2e-3, wd=0.05)
        self.supervisor = ConsensusSupervisor(self)
    def get_params(self):
        params = self.stem.get_params()
        for b in self.blocks: params.extend(b.get_params())
        params.extend(self.head_ln.get_params())
        params.extend(self.head.get_params())
        return params
    def forward(self, x, training=True):
        x = self.stem.forward(x)
        for b in self.blocks:
            b.training = training
            x = b.forward(x)
        x = self.head_ln.forward(x)
        return self.head.forward(x)
    def backward(self, grad):
        grad = self.head.backward(grad)
        grad = self.head_ln.backward(grad)
        for b in reversed(self.blocks):
            grad = b.backward(grad)
        self.stem.backward(grad)
    def train_step(self, x, y):
        logits = self.forward(x, training=True)
        shift_logits = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(shift_logits)
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        loss = -np.mean(np.sum(y * np.log(probs + 1e-12), axis=1))
        grad = (probs - y) / y.shape[0]
        self.backward(grad)
        self.optimizer.step()
        return loss
    def fit(self, x, y, epochs=100, batch_size=256):
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
            acc = self.evaluate(x[:1000], y[:1000])
            print(f"EP {epoch:03d} | LOSS: {avg_loss:.5f} | ACC: {acc:.4f} | LR: {self.optimizer.lr:.6f} | {time.time()-t0:.2f}s | {status}")
            if acc > 0.999: break
    def evaluate(self, x, y):
        logits = self.forward(x, training=False)
        return np.mean(np.argmax(logits, axis=1) == np.argmax(y, axis=1))

def generate_synthetic_data(n=15000, d=784, c=10):
    x = np.random.randn(n, d).astype(np.float32)
    # Create non-linear relationship
    w1 = np.random.randn(d, 512)
    w2 = np.random.randn(512, c)
    z = np.maximum(0, np.dot(x, w1))
    logits = np.dot(z, w2)
    y_idx = np.argmax(logits + 0.05 * np.random.randn(n, c), axis=1)
    y = np.eye(c)[y_idx].astype(np.float32)
    return (x - np.mean(x)) / (np.std(x) + 1e-8), y

if __name__ == "__main__":
    X, Y = generate_synthetic_data()
    model = OMEGA_ASI(784, 128, 10, blocks=4)
    model.fit(X, Y, epochs=100, batch_size=512)
    print("Evolutionary Cycle Complete.")
