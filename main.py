import numpy as np
import time
import os
import sys

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

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def get_params(self):
        return self.params

class Linear(Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
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

class GELU(Module):
    def forward(self, x):
        self.x = x
        self.tanh_out = np.tanh(0.7978845608 * (x + 0.044715 * np.power(x, 3)))
        return 0.5 * x * (1 + self.tanh_out)

    def backward(self, grad):
        x3 = np.power(self.x, 3)
        inner = 0.7978845608 * (self.x + 0.044715 * x3)
        sech2 = 1.0 / (np.cosh(inner) ** 2)
        deriv = 0.5 * (1 + self.tanh_out) + (0.5 * self.x * sech2 * 0.7978845608 * (1 + 3 * 0.044715 * self.x**2))
        return grad * deriv

class BatchNorm(Module):
    def __init__(self, dim, momentum=0.9, eps=1e-5):
        super().__init__()
        self.gamma = Parameter(np.ones((1, dim)), "gamma")
        self.beta = Parameter(np.zeros((1, dim)), "beta")
        self.params = [self.gamma, self.beta]
        self.running_mean = np.zeros((1, dim), dtype=np.float32)
        self.running_var = np.ones((1, dim), dtype=np.float32)
        self.momentum = momentum
        self.eps = eps

    def forward(self, x):
        if not self.training:
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            return self.gamma.data * x_hat + self.beta.data
        
        mu = np.mean(x, axis=0, keepdims=True)
        var = np.var(x, axis=0, keepdims=True)
        self.inv_std = 1.0 / np.sqrt(var + self.eps)
        self.x_centered = x - mu
        self.x_hat = self.x_centered * self.inv_std
        
        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
        self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        return self.gamma.data * self.x_hat + self.beta.data

    def backward(self, grad):
        N = grad.shape[0]
        self.gamma.grad = np.sum(grad * self.x_hat, axis=0, keepdims=True)
        self.beta.grad = np.sum(grad, axis=0, keepdims=True)
        
        dx_hat = grad * self.gamma.data
        dvar = np.sum(dx_hat * self.x_centered * -0.5 * (self.inv_std**3), axis=0, keepdims=True)
        dmu = np.sum(dx_hat * -self.inv_std, axis=0, keepdims=True) + dvar * np.mean(-2.0 * self.x_centered, axis=0, keepdims=True)
        
        return (dx_hat * self.inv_std) + (dvar * 2.0 * self.x_centered / N) + (dmu / N)

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
        self.bn1 = BatchNorm(dim)
        self.l1 = Linear(dim, dim * 2)
        self.act = GELU()
        self.l2 = Linear(dim * 2, dim)
        self.drop = Dropout(dropout_p)
        self.params = self.bn1.params + self.l1.params + self.l2.params

    def forward(self, x):
        self.residual = x
        out = self.bn1.forward(x)
        out = self.l1.forward(out)
        out = self.act.forward(out)
        out = self.l2.forward(out)
        out = self.drop.forward(out)
        return out + self.residual

    def backward(self, grad):
        dg = self.drop.backward(grad)
        dg = self.l2.backward(dg)
        dg = self.act.backward(dg)
        dg = self.l1.backward(dg)
        dg = self.bn1.backward(dg)
        return dg + grad

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
        self.groq_active = os.getenv("GROQ_API_KEY") is not None
        self.gemini_active = os.getenv("GEMINI_API_KEY") is not None

    def audit(self, loss):
        self.history.append(loss)
        if len(self.history) < 10: return "INITIALIZING"
        
        recent = self.history[-5:]
        trend = np.polyfit(np.arange(5), recent, 1)[0]
        
        # Redundant Logic Gates
        g_signal = self._groq_gate(trend, loss)
        m_signal = self._gemini_gate(trend, loss)
        
        if g_signal == "REDUCE" or m_signal == "REDUCE":
            self.model.optimizer.lr *= 0.5
            return "DEFLATIONARY_EVENT"
        if g_signal == "BOOST" and m_signal == "BOOST":
            self.model.optimizer.lr *= 1.05
            return "MOMENTUM_EXPANSION"
        return "STABLE"

    def _groq_gate(self, trend, loss):
        # High-speed heuristic simulation
        if trend > 0.05: return "REDUCE"
        if abs(trend) < 1e-4: return "BOOST"
        return "STABLE"

    def _gemini_gate(self, trend, loss):
        # Deep reasoning heuristic simulation
        if loss > 5.0 and trend > 0: return "REDUCE"
        if loss < 0.1: return "STABLE"
        if trend > -1e-5: return "BOOST"
        return "STABLE"

class OMEGA_ASI:
    def __init__(self, in_d, hid_d, out_d, num_blocks=4):
        self.layers = [
            Linear(in_d, hid_d),
            BatchNorm(hid_d),
            GELU()
        ]
        for _ in range(num_blocks):
            self.layers.append(ResidualBlock(hid_d))
        self.layers.append(Linear(hid_d, out_d))
        
        self.params = []
        for l in self.layers:
            self.params.extend(l.get_params())
            
        self.optimizer = AdamW(self.params, lr=2e-3, wd=0.01)
        self.supervisor = ConsensusSupervisor(self)

    def forward(self, x, training=True):
        for l in self.layers:
            l.training = training
            x = l.forward(x)
        return x

    def backward(self, grad):
        for l in reversed(self.layers):
            grad = l.backward(grad)

    def loss_fn(self, logits, y):
        shift = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(shift)
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        self.probs = probs
        return -np.mean(np.sum(y * np.log(probs + 1e-12), axis=1))

    def train_step(self, x, y):
        logits = self.forward(x, training=True)
        loss = self.loss_fn(logits, y)
        grad = (self.probs - y) / y.shape[0]
        self.backward(grad)
        self.optimizer.step()
        return loss

    def fit(self, x, y, epochs=100, batch_size=256):
        n = x.shape[0]
        print(f"ARCHITECT_INIT: SAMPLES={n} FEATURES={x.shape[1]}")
        for epoch in range(1, epochs + 1):
            idx = np.random.permutation(n)
            losses = []
            t0 = time.time()
            for i in range(0, n, batch_size):
                xb = x[idx[i:i+batch_size]]
                yb = y[idx[i:i+batch_size]]
                losses.append(self.train_step(xb, yb))
            
            avg_loss = np.mean(losses)
            status = self.supervisor.audit(avg_loss)
            
            if epoch % 2 == 0 or epoch == 1:
                acc = self.evaluate(x[:1000], y[:1000])
                dt = time.time() - t0
                print(f"E {epoch:03d} | L {avg_loss:.4f} | A {acc:.4f} | T {dt:.2f}s | {status}")

    def evaluate(self, x, y):
        logits = self.forward(x, training=False)
        return np.mean(np.argmax(logits, axis=1) == np.argmax(y, axis=1))

def generate_data(n=10000, d=784, c=10):
    x = np.random.randn(n, d).astype(np.float32)
    # Create non-linear relationship
    w = np.random.randn(d, c).astype(np.float32)
    logits = np.dot(x, w)
    y_idx = np.argmax(logits + 0.1 * np.random.randn(n, c), axis=1)
    y = np.eye(c)[y_idx].astype(np.float32)
    return x, y

if __name__ == "__main__":
    X, Y = generate_data()
    model = OMEGA_ASI(784, 512, 10, num_blocks=3)
    model.fit(X, Y, epochs=50, batch_size=128)
