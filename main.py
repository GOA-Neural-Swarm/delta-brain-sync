import numpy as np
import time

class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=0.01):
        self.lr, self.beta1, self.beta2, self.eps, self.wd = lr, betas[0], betas[1], eps, wd
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)
        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i]**2)
            params[i] -= lr_t * (self.m[i] / (np.sqrt(self.v[i]) + self.eps) + self.wd * params[i])

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
        self.dgamma = np.sum(dout * self.x_hat, axis=0, keepdims=True)
        self.dbeta = np.sum(dout, axis=0, keepdims=True)
        dx_hat = dout * self.gamma
        dx = (1. / D) * self.std_inv * (D * dx_hat - np.sum(dx_hat, axis=-1, keepdims=True) - self.x_hat * np.sum(dx_hat * self.x_hat, axis=-1, keepdims=True))
        return dx

    def get_params(self): return [self.gamma, self.beta]
    def get_grads(self): return [self.dgamma, self.dbeta]

class Linear:
    def __init__(self, in_d, out_d):
        scale = np.sqrt(2.0 / in_d)
        self.W = (np.random.randn(in_d, out_d) * scale).astype(np.float32)
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

class Swish:
    def forward(self, x):
        self.x = x
        self.sig = 1.0 / (1.0 + np.exp(-x))
        return x * self.sig

    def backward(self, dout):
        return dout * (self.sig + self.x * self.sig * (1.0 - self.sig))

class SovereignGatedBlock:
    def __init__(self, dim):
        self.ln = LayerNorm(dim)
        self.w_gemini_gate = Linear(dim, dim)
        self.w_gemini_val = Linear(dim, dim)
        self.w_groq_gate = Linear(dim, dim)
        self.w_groq_val = Linear(dim, dim)
        self.act_swish = Swish()
        self.w_out = Linear(dim, dim)

    def forward(self, x):
        self.res = x
        h = self.ln.forward(x)
        
        # Gemini Path (Swish-Gated)
        self.g1 = self.act_swish.forward(self.w_gemini_gate.forward(h))
        self.v1 = self.w_gemini_val.forward(h)
        path_gemini = self.g1 * self.v1
        
        # Groq Path (Tanh-Gated Redundancy)
        self.g2 = np.tanh(self.w_groq_gate.forward(h))
        self.v2 = self.w_groq_val.forward(h)
        path_groq = self.g2 * self.v2
        
        self.combined = path_gemini + path_groq
        return self.w_out.forward(self.combined) + x

    def backward(self, dout):
        dout_out = self.w_out.backward(dout)
        
        # Gemini Backward
        dg1 = dout_out * self.v1
        dv1 = dout_out * self.g1
        dh_gemini = self.w_gemini_gate.backward(self.act_swish.backward(dg1)) + self.w_gemini_val.backward(dv1)
        
        # Groq Backward
        dg2 = dout_out * self.v2
        dv2 = dout_out * self.g2
        dtanh = dg2 * (1.0 - self.g2**2)
        dh_groq = self.w_groq_gate.backward(dtanh) + self.w_groq_val.backward(dv2)
        
        return self.ln.backward(dh_gemini + dh_groq) + dout

    def get_layers(self):
        return [self.ln, self.w_gemini_gate, self.w_gemini_val, self.w_groq_gate, self.w_groq_val, self.w_out]

class OMEGA_ASI_Engine:
    def __init__(self, in_d=784, h_d=256, out_d=10, num_blocks=4):
        self.stem = Linear(in_d, h_d)
        self.blocks = [SovereignGatedBlock(h_d) for _ in range(num_blocks)]
        self.head_ln = LayerNorm(h_d)
        self.head = Linear(h_d, out_d)

        self.layers = [self.stem]
        for b in self.blocks: self.layers.extend(b.get_layers())
        self.layers.extend([self.head_ln, self.head])

        self.params = []
        for l in self.layers:
            if hasattr(l, 'get_params'): self.params.extend(l.get_params())
        self.optimizer = AdamW(self.params, lr=1e-3, wd=0.05)

    def forward(self, x):
        x = self.stem.forward(x)
        for b in self.blocks: x = b.forward(x)
        return self.head.forward(self.head_ln.forward(x))

    def backward(self, dout):
        dout = self.head.backward(dout)
        dout = self.head_ln.backward(dout)
        for b in reversed(self.blocks): dout = b.backward(dout)
        self.stem.backward(dout)

        grads = []
        for l in self.layers:
            if hasattr(l, 'get_grads'): grads.extend(l.get_grads())
        self.optimizer.step(self.params, grads)

def execute_evolution():
    np.random.seed(42)
    N, D, C = 2048, 784, 10
    X = np.random.randn(N, D).astype(np.float32)
    Y = np.random.randint(0, C, N)

    model = OMEGA_ASI_Engine(in_d=D, h_d=128, out_d=C, num_blocks=3)
    batch_size = 64
    epochs = 100

    print("--- OMEGA-ASI: HIGH-PERFORMANCE EVOLUTION START ---")
    start = time.time()

    for epoch in range(1, epochs + 1):
        indices = np.random.permutation(N)
        total_loss, total_acc = 0, 0

        for i in range(0, N, batch_size):
            idx = indices[i:i+batch_size]
            xb, yb = X[idx], Y[idx]

            logits = model.forward(xb)
            
            # Stable Softmax
            shift_logits = logits - np.max(logits, axis=1, keepdims=True)
            exps = np.exp(shift_logits)
            probs = exps / np.sum(exps, axis=1, keepdims=True)

            loss = -np.mean(np.log(probs[range(len(yb)), yb] + 1e-10))
            acc = np.mean(np.argmax(probs, axis=1) == yb)

            d_logits = probs.copy()
            d_logits[range(len(yb)), yb] -= 1
            d_logits /= len(yb)
            
            model.backward(d_logits)

            total_loss += loss * (len(yb) / N)
            total_acc += acc * (len(yb) / N)

        if epoch % 10 == 0 or epoch == 1:
            print(f"EPOCH:{epoch:03d} | LOSS:{total_loss:.4f} | ACC:{total_acc:.4f} | T:{time.time()-start:.2f}s")

    print(f"--- EVOLUTION COMPLETE | FINAL ACCURACY: {total_acc:.4f} ---")

if __name__ == "__main__":
    execute_evolution()
