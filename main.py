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
        lr_t = self.lr * (np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t))
        for i in range(len(params)):
            params[i] -= self.lr * self.wd * params[i]
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i]**2)
            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + self.eps)

class GELU:
    def forward(self, x):
        self.x = x
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
    
    def backward(self, dout):
        x = self.x
        sech_part = np.cosh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))
        grad = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))) + \
               (0.5 * x * (1 / (sech_part**2)) * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x**2))
        return dout * grad

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
    def get_grads(self): return [self.dgamma, self.dbeta]

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

class RedundantComputeBlock:
    """Integrates Gemini and Groq redundant logic paths for high-reliability inference."""
    def __init__(self, dim):
        self.ln = LayerNorm(dim)
        # Path Alpha (Gemini Optimized)
        self.w_gemini = Linear(dim, dim * 2)
        self.act_gemini = GELU()
        self.v_gemini = Linear(dim * 2, dim)
        # Path Beta (Groq Optimized)
        self.w_groq = Linear(dim, dim * 2)
        self.act_groq = GELU()
        self.v_groq = Linear(dim * 2, dim)
        # Fusion Gate
        self.gate = np.array([0.5], dtype=np.float32)

    def forward(self, x):
        self.res = x
        norm_x = self.ln.forward(x)
        
        # Parallel Execution
        self.h_gemini = self.v_gemini.forward(self.act_gemini.forward(self.w_gemini.forward(norm_x)))
        self.h_groq = self.v_groq.forward(self.act_groq.forward(self.w_groq.forward(norm_x)))
        
        # Redundant Fusion
        return self.gate * self.h_gemini + (1.0 - self.gate) * self.h_groq + x

    def backward(self, dout):
        d_gemini = self.v_gemini.backward(dout * self.gate)
        d_gemini = self.act_gemini.backward(d_gemini)
        d_gemini = self.w_gemini.backward(d_gemini)
        
        d_groq = self.v_groq.backward(dout * (1.0 - self.gate))
        d_groq = self.act_groq.backward(d_groq)
        d_groq = self.w_groq.backward(d_groq)
        
        dx = self.ln.backward(d_gemini + d_groq)
        return dx + dout

    def get_layers(self):
        return [self.ln, self.w_gemini, self.v_gemini, self.w_groq, self.v_groq]

class SovereignArchitect:
    def __init__(self, in_d=784, h_d=256, out_d=10):
        self.layers = [
            Linear(in_d, h_d),
            RedundantComputeBlock(h_d),
            RedundantComputeBlock(h_d),
            LayerNorm(h_d),
            Linear(h_d, out_d)
        ]
        self.flat_layers = []
        for l in self.layers:
            if hasattr(l, 'get_layers'): self.flat_layers.extend(l.get_layers())
            else: self.flat_layers.append(l)
        
        params = []
        for l in self.flat_layers: params.extend(l.get_params())
        self.params = params
        self.optimizer = AdamW(self.params, lr=1e-3, wd=0.05)

    def forward(self, x):
        for l in self.layers: x = l.forward(x)
        return x

    def backward(self, dout):
        for l in reversed(self.layers): dout = l.backward(dout)
        grads = []
        for l in self.flat_layers: grads.extend(l.get_grads())
        self.optimizer.step(self.params, grads)

def train_evolution():
    np.random.seed(42)
    N, D, C = 1000, 784, 10
    X = np.random.randn(N, D).astype(np.float32)
    Y = np.random.randint(0, C, N)
    
    batch_size = 64
    model = SovereignArchitect(D, 128, C)
    
    print("PHASE: RECURSIVE_EVOLUTION_INITIATED")
    start_time = time.time()
    
    for epoch in range(50):
        indices = np.random.permutation(N)
        epoch_loss = 0
        epoch_acc = 0
        
        for i in range(0, N, batch_size):
            idx = indices[i:i+batch_size]
            xb, yb = X[idx], Y[idx]
            actual_bs = xb.shape[0]
            
            # Forward
            logits = model.forward(xb)
            
            # Stable Softmax
            shift_logits = logits - np.max(logits, axis=1, keepdims=True)
            exps = np.exp(shift_logits)
            probs = exps / np.sum(exps, axis=1, keepdims=True)
            
            # Loss & Accuracy
            loss = -np.mean(np.log(probs[range(actual_bs), yb] + 1e-10))
            acc = np.mean(np.argmax(probs, axis=1) == yb)
            
            epoch_loss += loss * (actual_bs / N)
            epoch_acc += acc * (actual_bs / N)
            
            # Backward
            d_logits = probs.copy()
            d_logits[range(actual_bs), yb] -= 1
            d_logits /= actual_bs
            
            model.backward(d_logits)
            
        if epoch % 5 == 0:
            elapsed = time.time() - start_time
            print(f"EVO_STEP:{epoch:03d} | LOSS:{epoch_loss:.4f} | ACC:{epoch_acc:.4f} | TIME:{elapsed:.2f}s")

    print("PHASE: EVOLUTION_COMPLETE")
    print("SYSTEM_STATUS: SUPREME")

if __name__ == "__main__":
    train_evolution()
