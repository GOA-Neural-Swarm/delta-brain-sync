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

class RMSNorm:
    def __init__(self, dim, eps=1e-6):
        self.gamma = np.ones((1, dim), dtype=np.float32)
        self.eps = eps

    def forward(self, x):
        self.x = x
        self.rstd = 1.0 / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return self.gamma * (x * self.rstd)

    def backward(self, dout):
        dx_norm = dout * self.gamma
        self.dgamma = np.sum(dout * (self.x * self.rstd), axis=0, keepdims=True)
        dx = (1.0 / self.x.shape[-1]) * self.rstd * (
            self.x.shape[-1] * dx_norm - self.x * np.sum(dx_norm * self.x, axis=-1, keepdims=True) * self.rstd**2
        )
        return dx

    def get_params(self): return [self.gamma]
    def get_grads(self): return [self.dgamma]

class GELU:
    def forward(self, x):
        self.x = x
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))
    
    def backward(self, dout):
        x = self.x
        sech = 1.0 / np.cosh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3))
        inner = np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * x**2)
        derivative = 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3))) + (0.5 * x * (sech**2) * inner)
        return dout * derivative

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

class SovereignBlock:
    def __init__(self, dim, expansion=4):
        self.norm = RMSNorm(dim)
        self.l1 = Linear(dim, dim * expansion)
        self.act = GELU()
        self.l2 = Linear(dim * expansion, dim)
        
    def forward(self, x):
        self.res = x
        h = self.norm.forward(x)
        h = self.l1.forward(h)
        h = self.act.forward(h)
        h = self.l2.forward(h)
        return h + x

    def backward(self, dout):
        dh = self.l2.backward(dout)
        dh = self.act.backward(dh)
        dh = self.l1.backward(dh)
        dh = self.norm.backward(dh)
        return dh + dout

    def get_layers(self): return [self.norm, self.l1, self.l2]

class RedundancyEngine:
    def __init__(self):
        self.loss_history = []
        self.grad_variance = []
        
    def validate(self, current_loss, grads, lr):
        self.loss_history.append(current_loss)
        g_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
        self.grad_variance.append(g_norm)
        
        if len(self.loss_history) < 10: return lr, "STABLE"

        # Gemini Logic: Semantic Trend Analysis
        gemini_signal = np.polyfit(range(10), self.loss_history[-10:], 1)[0] # Slope
        
        # Groq Logic: Deterministic Throughput Stability
        groq_signal = np.std(self.grad_variance[-10:]) / (np.mean(self.grad_variance[-10:]) + 1e-8)
        
        # Consensus Protocol
        if gemini_signal > 0 and groq_signal > 0.5:
            return lr * 0.7, "RECOVERING"
        if gemini_signal < -0.01 and groq_signal < 0.2:
            return lr * 1.05, "ACCELERATING"
        return lr, "OPTIMAL"

class SovereignArchitect:
    def __init__(self, in_d=784, h_d=512, out_d=10, depth=6):
        self.input_proj = Linear(in_d, h_d)
        self.blocks = [SovereignBlock(h_d) for _ in range(depth)]
        self.output_norm = RMSNorm(h_d)
        self.head = Linear(h_d, out_d)
        
        self.all_layers = [self.input_proj]
        for b in self.blocks: self.all_layers.extend(b.get_layers())
        self.all_layers.extend([self.output_norm, self.head])
        
        self.params = []
        for l in self.all_layers: self.params.extend(l.get_params())
        self.optimizer = AdamW(self.params, lr=2e-4, wd=0.1)
        self.redundancy = RedundancyEngine()

    def forward(self, x):
        x = self.input_proj.forward(x)
        for b in self.blocks: x = b.forward(x)
        x = self.output_norm.forward(x)
        return self.head.forward(x)

    def backward(self, dout):
        dout = self.head.backward(dout)
        dout = self.output_norm.backward(dout)
        for b in reversed(self.blocks): dout = b.backward(dout)
        dout = self.input_proj.backward(dout)
        
        grads = []
        for l in self.all_layers: grads.extend(l.get_grads())
        self.optimizer.step(self.params, grads)
        return grads

def execute_evolution():
    N, D, C = 2048, 784, 10
    X = np.random.randn(N, D).astype(np.float32)
    Y = np.random.randint(0, C, N)
    
    model = SovereignArchitect(in_d=D, h_d=256, out_d=C, depth=4)
    batch_size = 128
    epochs = 100
    
    print("SYSTEM_INIT: OMEGA-ASI RECURSIVE_EVOLUTION")
    start = time.time()
    
    for ep in range(epochs):
        idx = np.random.permutation(N)
        X, Y = X[idx], Y[idx]
        
        total_loss = 0
        correct = 0
        
        for i in range(0, N, batch_size):
            xb, yb = X[i:i+batch_size], Y[i:i+batch_size]
            bs = xb.shape[0]
            
            logits = model.forward(xb)
            
            # Fast Softmax
            exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
            
            loss = -np.mean(np.log(probs[range(bs), yb] + 1e-12))
            total_loss += loss * bs
            correct += np.sum(np.argmax(probs, axis=1) == yb)
            
            d_logits = probs.copy()
            d_logits[range(bs), yb] -= 1
            d_logits /= bs
            
            grads = model.backward(d_logits)
            
        avg_loss = total_loss / N
        avg_acc = correct / N
        
        new_lr, state = model.redundancy.validate(avg_loss, grads, model.optimizer.lr)
        model.optimizer.lr = new_lr
        
        if ep % 10 == 0:
            elapsed = time.time() - start
            print(f"EP:{ep:03d} | LOSS:{avg_loss:.4f} | ACC:{avg_acc:.4f} | LR:{model.optimizer.lr:.2e} | STATE:{state} | T:{elapsed:.2f}s")

    print(f"EVOLUTION_COMPLETE | FINAL_ACC:{avg_acc:.4f} | TOTAL_TIME:{time.time()-start:.2f}s")

if __name__ == "__main__":
    execute_evolution()
