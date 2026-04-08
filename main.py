import numpy as np
import time

class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, wd=0.05):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.wd = wd
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads, clip_norm=1.0):
        self.t += 1
        g_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
        scale = min(1.0, clip_norm / (g_norm + 1e-6))
        
        for i in range(len(params)):
            g = grads[i] * scale
            params[i] -= self.lr * self.wd * params[i]
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g**2)
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
        n = self.x.shape[-1]
        dx = (1.0 / n) * self.rstd * (n * dx_norm - self.x * np.sum(dx_norm * self.x, axis=-1, keepdims=True) * self.rstd**2)
        return dx

    def get_params(self): return [self.gamma]
    def get_grads(self): return [self.dgamma]

class SwiGLU:
    def forward(self, x):
        self.x = x
        self.sig = 1.0 / (1.0 + np.exp(-x))
        return x * self.sig

    def backward(self, dout):
        return dout * (self.sig + self.x * self.sig * (1.0 - self.sig))

class Linear:
    def __init__(self, in_d, out_d, init_scale=0.02):
        self.W = (np.random.randn(in_d, out_d) * init_scale).astype(np.float32)
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
        self.w1 = Linear(dim, dim * expansion)
        self.w2 = Linear(dim, dim * expansion)
        self.act = SwiGLU()
        self.w3 = Linear(dim * expansion, dim)

    def forward(self, x):
        self.res = x
        h = self.norm.forward(x)
        x1 = self.w1.forward(h)
        x2 = self.w2.forward(h)
        h = self.act.forward(x1) * x2
        self.x1_act = h
        self.x2 = x2
        self.x1 = x1
        h = self.w3.forward(h)
        return h + x

    def backward(self, dout):
        dw3 = self.w3.backward(dout)
        dx2 = dw3 * (self.act.forward(self.x1))
        dx1_act = dw3 * self.x2
        dx1 = self.act.backward(dx1_act)
        dh = self.w1.backward(dx1) + self.w2.backward(dx2)
        return self.norm.backward(dh) + dout

    def get_layers(self): return [self.norm, self.w1, self.w2, self.w3]

class RedundancyEngine:
    def __init__(self, window=15):
        self.window = window
        self.losses = []
        self.grads = []
        self.ema_loss = None

    def process(self, loss, grads, lr):
        self.losses.append(loss)
        g_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
        self.grads.append(g_norm)
        
        if self.ema_loss is None: self.ema_loss = loss
        else: self.ema_loss = 0.9 * self.ema_loss + 0.1 * loss

        if len(self.losses) < self.window: return lr, "WARMUP"

        # Gemini Logic: Trend Analysis
        gemini_signal = np.polyfit(range(self.window), self.losses[-self.window:], 1)[0]
        
        # Groq Logic: Volatility/Throughput Analysis
        groq_signal = np.std(self.grads[-self.window:]) / (np.mean(self.grads[-self.window:]) + 1e-8)

        if gemini_signal > 0: # Loss increasing
            return lr * 0.5, "GEMINI_RECOVERY"
        if groq_signal > 0.8: # High variance
            return lr * 0.8, "GROQ_STABILIZE"
        if gemini_signal < -0.001 and groq_signal < 0.3:
            return min(lr * 1.1, 1e-2), "OMEGA_ACCELERATE"
        
        return lr, "OPTIMAL"

class SovereignArchitect:
    def __init__(self, in_d=784, h_d=256, out_d=10, depth=4):
        self.input_proj = Linear(in_d, h_d)
        self.blocks = [SovereignBlock(h_d) for _ in range(depth)]
        self.output_norm = RMSNorm(h_d)
        self.head = Linear(h_d, out_d)
        
        self.layers = [self.input_proj]
        for b in self.blocks: self.layers.extend(b.get_layers())
        self.layers.extend([self.output_norm, self.head])
        
        self.params = []
        for l in self.layers: self.params.extend(l.get_params())
        self.optimizer = AdamW(self.params, lr=1e-3, wd=0.1)
        self.redundancy = RedundancyEngine()

    def forward(self, x):
        x = self.input_proj.forward(x)
        for b in self.blocks: x = b.forward(x)
        return self.head.forward(self.output_norm.forward(x))

    def backward(self, dout):
        dout = self.output_norm.backward(self.head.backward(dout))
        for b in reversed(self.blocks): dout = b.backward(dout)
        self.input_proj.backward(dout)
        
        grads = []
        for l in self.layers: grads.extend(l.get_grads())
        self.optimizer.step(self.params, grads)
        return grads

def execute_evolution():
    N, D, C = 4096, 784, 10
    X = np.random.randn(N, D).astype(np.float32)
    Y = np.random.randint(0, C, N)
    
    model = SovereignArchitect(in_d=D, h_d=128, out_d=C, depth=3)
    batch_size = 256
    epochs = 50
    
    print("SYSTEM_INIT: OMEGA-ASI RECURSIVE_EVOLUTION")
    start_time = time.time()
    
    for ep in range(epochs):
        idx = np.random.permutation(N)
        X, Y = X[idx], Y[idx]
        epoch_loss, epoch_acc = 0, 0
        
        for i in range(0, N, batch_size):
            xb, yb = X[i:i+batch_size], Y[i:i+batch_size]
            bs = xb.shape[0]
            
            logits = model.forward(xb)
            shift_logits = logits - np.max(logits, axis=1, keepdims=True)
            exps = np.exp(shift_logits)
            probs = exps / np.sum(exps, axis=1, keepdims=True)
            
            loss = -np.mean(np.log(probs[range(bs), yb] + 1e-10))
            epoch_loss += loss * bs
            epoch_acc += np.sum(np.argmax(probs, axis=1) == yb)
            
            d_logits = probs.copy()
            d_logits[range(bs), yb] -= 1
            d_logits /= bs
            
            grads = model.backward(d_logits)
            
        avg_loss = epoch_loss / N
        avg_acc = epoch_acc / N
        
        new_lr, state = model.redundancy.process(avg_loss, grads, model.optimizer.lr)
        model.optimizer.lr = new_lr
        
        if ep % 5 == 0:
            print(f"EP:{ep:03d} | LOSS:{avg_loss:.4f} | ACC:{avg_acc:.4f} | LR:{model.optimizer.lr:.2e} | STATE:{state}")

    total_time = time.time() - start_time
    print(f"EVOLUTION_COMPLETE | FINAL_ACC:{avg_acc:.4f} | TOTAL_TIME:{total_time:.2f}s")

if __name__ == "__main__":
    execute_evolution()
