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

class ResidualBlock(Module):
    def __init__(self, dim, dropout_p=0.1):
        super().__init__()
        self.ln1 = LayerNorm(dim)
        self.l1 = Linear(dim, dim * 4)
        self.act = GELU()
        self.l2 = Linear(dim * 4, dim)
        self.drop = Dropout(dropout_p)
        self.params = self.ln1.params + self.l1.params + self.l2.params
    def forward(self, x):
        self.res = x
        out = self.ln1.forward(x)
        out = self.l1.forward(out)
        out = self.act.forward(out)
        out = self.l2.forward(out)
        return self.drop.forward(out) + self.res
    def backward(self, grad):
        dx = self.drop.backward(grad)
        dx = self.l2.backward(dx)
        dx = self.act.backward(dx)
        dx = self.l1.backward(dx)
        dx = self.ln1.backward(dx)
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
        self.grad_variance = []
        self.ema_loss = None
    def audit(self, loss, params):
        self.loss_history.append(loss)
        if self.ema_loss is None: self.ema_loss = loss
        else: self.ema_loss = 0.9 * self.ema_loss + 0.1 * loss
        
        all_grads = np.concatenate([p.grad.flatten() for p in params])
        gv = np.var(all_grads)
        self.grad_variance.append(gv)
        
        if len(self.loss_history) < 5: return "WARMUP"
        
        # Groq Logic: Gradient Signal-to-Noise Ratio (SNR)
        # High variance relative to mean suggests unstable updates
        groq_signal = "STABLE"
        if gv > np.mean(self.grad_variance[-10:]) * 2.0: groq_signal = "REDUCE"
        elif gv < np.mean(self.grad_variance[-10:]) * 0.5: groq_signal = "BOOST"
        
        # Gemini Logic: Loss Trajectory & Curvature
        gemini_signal = "STABLE"
        recent = self.loss_history[-5:]
        slope = np.polyfit(np.arange(5), recent, 1)[0]
        if slope > 0: gemini_signal = "REDUCE"
        elif abs(slope) < 1e-5: gemini_signal = "BOOST"
        
        if groq_signal == "REDUCE" or gemini_signal == "REDUCE":
            self.model.optimizer.lr *= 0.7
            return f"DEFLATE({groq_signal[0]}{gemini_signal[0]})"
        if groq_signal == "BOOST" and gemini_signal == "BOOST":
            self.model.optimizer.lr = min(self.model.optimizer.lr * 1.1, 5e-3)
            return f"EXPAND({groq_signal[0]}{gemini_signal[0]})"
        return "OPTIMAL"

class OMEGA_ASI:
    def __init__(self, in_d, hid_d, out_d, blocks=4):
        self.layers = [Linear(in_d, hid_d), LayerNorm(hid_d), GELU()]
        for _ in range(blocks):
            self.layers.append(ResidualBlock(hid_d))
        self.layers.append(Linear(hid_d, out_d))
        self.params = []
        for l in self.layers: self.params.extend(l.get_params())
        self.optimizer = AdamW(self.params, lr=1e-3, wd=0.02)
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
        max_l = np.max(logits, axis=1, keepdims=True)
        exps = np.exp(logits - max_l)
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        loss = -np.mean(np.sum(y * np.log(probs + 1e-10), axis=1))
        grad = (probs - y) / y.shape[0]
        self.backward(grad)
        self.optimizer.step()
        return loss
    def fit(self, x, y, epochs=50, batch_size=256):
        n = x.shape[0]
        for epoch in range(1, epochs + 1):
            idx = np.random.permutation(n)
            losses = []
            t0 = time.time()
            for i in range(0, n, batch_size):
                xb, yb = x[idx[i:i+batch_size]], y[idx[i:i+batch_size]]
                losses.append(self.train_step(xb, yb))
            avg_loss = np.mean(losses)
            status = self.supervisor.audit(avg_loss, self.params)
            acc = self.evaluate(x[:1000], y[:1000])
            print(f"EP {epoch:02d} | LOSS: {avg_loss:.4f} | ACC: {acc:.4f} | LR: {self.optimizer.lr:.5f} | {time.time()-t0:.2f}s | {status}")
            if acc > 0.995: break
    def evaluate(self, x, y):
        logits = self.forward(x, training=False)
        return np.mean(np.argmax(logits, axis=1) == np.argmax(y, axis=1))

def get_data(n=10000, d=784, c=10):
    x = np.random.randn(n, d).astype(np.float32)
    w = np.random.randn(d, c).astype(np.float32)
    y_idx = np.argmax(np.dot(x, w) + 0.1 * np.random.randn(n, c), axis=1)
    y = np.eye(c)[y_idx].astype(np.float32)
    return (x - np.mean(x)) / (np.std(x) + 1e-8), y

if __name__ == "__main__":
    X, Y = get_data()
    model = OMEGA_ASI(784, 256, 10, blocks=4)
    model.fit(X, Y, epochs=100, batch_size=512)
