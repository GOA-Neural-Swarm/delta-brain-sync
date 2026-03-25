
import numpy as np
import time
import os

class AdamW:
    def __init__(self, params_ref, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p) for p in params_ref]
        self.v = [np.zeros_like(p) for p in params_ref]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        lr_t = self.lr * (np.sqrt(1.0 - self.betas[1]**self.t) / (1.0 - self.betas[0]**self.t))
        for i in range(len(params)):
            params[i] -= self.lr * self.weight_decay * params[i]
            self.m[i] = self.betas[0] * self.m[i] + (1.0 - self.betas[0]) * grads[i]
            self.v[i] = self.betas[1] * self.v[i] + (1.0 - self.betas[1]) * (grads[i]**2)
            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + self.eps)

class Layer:
    def __init__(self):
        self.params = []
        self.grads = []
        self.training = True

    def forward(self, x): raise NotImplementedError
    def backward(self, grad): raise NotImplementedError

class Linear(Layer):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        self.w = np.random.uniform(-limit, limit, (in_dim, out_dim))
        self.b = np.zeros((1, out_dim))
        self.params = [self.w, self.b]
        self.x = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.w) + self.b

    def backward(self, grad):
        dw = np.dot(self.x.T, grad)
        db = np.sum(grad, axis=0, keepdims=True)
        self.grads = [dw, db]
        return np.dot(grad, self.w.T)

class BatchNorm(Layer):
    def __init__(self, dim, momentum=0.9, eps=1e-5):
        super().__init__()
        self.gamma = np.ones((1, dim))
        self.beta = np.zeros((1, dim))
        self.params = [self.gamma, self.beta]
        self.running_mean = np.zeros((1, dim))
        self.running_var = np.ones((1, dim))
        self.momentum = momentum
        self.eps = eps
        self.cache = None

    def forward(self, x):
        if not self.training:
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            return self.gamma * x_hat + self.beta

        mu = np.mean(x, axis=0, keepdims=True)
        var = np.var(x, axis=0, keepdims=True)
        x_hat = (x - mu) / np.sqrt(var + self.eps)
        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
        self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        self.cache = (x, x_hat, mu, var)
        return self.gamma * x_hat + self.beta

    def backward(self, grad):
        x, x_hat, mu, var = self.cache
        N = grad.shape[0]
        dgamma = np.sum(grad * x_hat, axis=0, keepdims=True)
        dbeta = np.sum(grad, axis=0, keepdims=True)
        dx_hat = grad * self.gamma
        dvar = np.sum(dx_hat * (x - mu) * -0.5 * (var + self.eps)**-1.5, axis=0, keepdims=True)
        dmu = np.sum(dx_hat * -1 / np.sqrt(var + self.eps), axis=0, keepdims=True) + dvar * np.mean(-2 * (x - mu), axis=0, keepdims=True)
        dx = dx_hat / np.sqrt(var + self.eps) + dvar * 2 * (x - mu) / N + dmu / N
        self.grads = [dgamma, dbeta]
        return dx

class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask

class Dropout(Layer):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, x):
        if not self.training: return x
        self.mask = (np.random.rand(*x.shape) > self.p) / (1.0 - self.p)
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask

class SoftmaxCrossEntropy:
    def __call__(self, logits, y):
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exps / np.sum(exps, axis=1, keepdims=True)
        return -np.mean(np.sum(y * np.log(self.probs + 1e-12), axis=1))

    def backward(self, y):
        return (self.probs - y) / y.shape[0]

class AdaptiveSupervisor:
    def __init__(self):
        self.history = []
        self.groq_active = os.getenv("GROQ_API_KEY") is not None
        self.gemini_active = os.getenv("GEMINI_API_KEY") is not None

    def analyze_telemetry(self, loss, optimizer):
        self.history.append(loss)
        if len(self.history) < 3: return "INITIALIZING"

        slope = self.history[-1] - self.history[-2]

        if self.groq_active and slope > 0:
            optimizer.lr *= 0.7
            return "GROQ_RECOVERY_DEFLATION"

        if self.gemini_active and abs(slope) < 1e-4:
            optimizer.lr *= 1.2
            return "GEMINI_MOMENTUM_INJECTION"

        if slope > 0.1:
            optimizer.lr *= 0.5
            return "LOCAL_STABILIZATION"

        return "STABLE_EVOLUTION"

class OMEGA_ASI:
    def __init__(self, input_dim, hidden_dims, output_dim):
        self.layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims)-1):
            self.layers.append(Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                self.layers.append(BatchNorm(dims[i+1]))
                self.layers.append(ReLU())
                self.layers.append(Dropout(0.1))

        self.loss_fn = SoftmaxCrossEntropy()
        all_params = []
        for l in self.layers:
            all_params.extend(l.params)
        self.optimizer = AdamW(all_params, lr=2e-3)
        self.supervisor = AdaptiveSupervisor()

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
        loss = self.loss_fn(logits, y)
        grad = self.loss_fn.backward(y)
        self.backward(grad)

        all_params, all_grads = [], []
        for l in self.layers:
            if l.params:
                all_params.extend(l.params)
                all_grads.extend(l.grads)
        self.optimizer.step(all_params, all_grads)
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
            protocol = self.supervisor.analyze_telemetry(avg_loss, self.optimizer)
            dt = time.time() - t0

            if epoch % 5 == 0 or epoch == 1:
                acc = self.evaluate(x[:1000], y[:1000])
                print(f"[EVO {epoch:03d}] Loss: {avg_loss:.5f} | Acc: {acc:.4f} | {dt:.3f}s | Protocol: {protocol}")

    def evaluate(self, x, y):
        logits = self.forward(x, training=False)
        preds = np.argmax(logits, axis=1)
        labels = np.argmax(y, axis=1)
        return np.mean(preds == labels)

def generate_data(n=10000, d=784, c=10):
    x = np.random.randn(n, d).astype(np.float32)
    w_true = np.random.randn(d, c)
    logits = np.dot(x, w_true)
    y_idx = np.argmax(logits + np.random.normal(0, 0.1, (n, c)), axis=1)
    y = np.zeros((n, c))
    y[np.arange(n), y_idx] = 1
    return x, y

if __name__ == "__main__":
    print("--- OMEGA-ASI: RECURSIVE ARCHITECT ACTIVATED ---")
    X, Y = generate_data()
    model = OMEGA_ASI(784, [512, 256], 10)
    model.fit(X, Y, epochs=50, batch_size=256)
    print("--- EVOLUTION CYCLE COMPLETE ---")
