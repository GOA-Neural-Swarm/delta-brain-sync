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

class LayerNorm:
    def __init__(self, dim, eps=1e-6):
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

class SiLU:
    def forward(self, x):
        self.x = x
        self.sig = 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
        return x * self.sig
    def backward(self, dout):
        return dout * (self.sig * (1.0 + self.x * (1.0 - self.sig)))

class Linear:
    def __init__(self, in_d, out_d):
        limit = np.sqrt(6.0 / (in_d + out_d))
        self.W = np.random.uniform(-limit, limit, (in_d, out_d)).astype(np.float32)
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

class GeminiLogicPath:
    def __init__(self, dim):
        self.ln = LayerNorm(dim)
        self.l1 = Linear(dim, dim * 2)
        self.act = SiLU()
        self.l2 = Linear(dim * 2, dim)
    def forward(self, x):
        h = self.ln.forward(x)
        h = self.l1.forward(h)
        h = self.act.forward(h)
        return self.l2.forward(h)
    def backward(self, dout):
        dh = self.l2.backward(dout)
        dh = self.act.backward(dh)
        dh = self.l1.backward(dh)
        return self.ln.backward(dh)
    def get_layers(self): return [self.ln, self.l1, self.l2]

class GroqLogicPath:
    def __init__(self, dim):
        self.ln = LayerNorm(dim)
        self.l1 = Linear(dim, dim)
        self.act = SiLU()
    def forward(self, x):
        h = self.ln.forward(x)
        h = self.l1.forward(h)
        return self.act.forward(h)
    def backward(self, dout):
        dh = self.act.backward(dout)
        dh = self.l1.backward(dh)
        return self.ln.backward(dh)
    def get_layers(self): return [self.ln, self.l1]

class RedundantConsensusBlock:
    def __init__(self, dim):
        self.gemini = GeminiLogicPath(dim)
        self.groq = GroqLogicPath(dim)
        self.gate = Linear(dim, 2)
        
    def forward(self, x):
        self.res = x
        g_out = self.gemini.forward(x)
        q_out = self.groq.forward(x)
        
        gate_logits = self.gate.forward(x)
        ex = np.exp(gate_logits - np.max(gate_logits, axis=1, keepdims=True))
        self.probs = ex / np.sum(ex, axis=1, keepdims=True)
        
        self.g_out, self.q_out = g_out, q_out
        return self.probs[:, 0:1] * g_out + self.probs[:, 1:2] * q_out + x

    def backward(self, dout):
        dg_out = dout * self.probs[:, 0:1]
        dq_out = dout * self.probs[:, 1:2]
        
        dprobs = np.zeros_like(self.probs)
        dprobs[:, 0] = np.sum(dout * self.g_out, axis=1)
        dprobs[:, 1] = np.sum(dout * self.q_out, axis=1)
        
        # Softmax backward
        dgate_logits = self.probs * (dprobs - np.sum(self.probs * dprobs, axis=1, keepdims=True))
        dgate = self.gate.backward(dgate_logits)
        
        dgemini = self.gemini.backward(dg_out)
        dgroq = self.groq.backward(dq_out)
        
        return dgemini + dgroq + dgate + dout

    def get_layers(self):
        layers = []
        layers.extend(self.gemini.get_layers())
        layers.extend(self.groq.get_layers())
        layers.append(self.gate)
        return layers

class OMEGA_ASI_Engine:
    def __init__(self, in_d=784, h_d=256, out_d=10):
        self.layers = [
            Linear(in_d, h_d),
            RedundantConsensusBlock(h_d),
            RedundantConsensusBlock(h_d),
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
    N, D, C = 1000, 784, 10
    X = np.random.randn(N, D).astype(np.float32)
    Y = np.random.randint(0, C, N)
    
    batch_size = 64
    model = OMEGA_ASI_Engine(D, 128, C)
    
    print("PHASE: RECURSIVE_EVOLUTION_INITIATED")
    start_time = time.time()
    
    for epoch in range(50):
        indices = np.random.permutation(N)
        epoch_loss = 0
        epoch_acc = 0
        
        for i in range(0, N, batch_size):
            idx = indices[i:i+batch_size]
            xb, yb = X[idx], Y[idx]
            curr_bs = xb.shape[0]
            
            logits = model.forward(xb)
            
            # Stable Softmax
            shift_logits = logits - np.max(logits, axis=1, keepdims=True)
            ex = np.exp(shift_logits)
            probs = ex / np.sum(ex, axis=1, keepdims=True)
            
            loss = -np.mean(np.log(probs[range(curr_bs), yb] + 1e-10))
            acc = np.mean(np.argmax(probs, axis=1) == yb)
            
            d_logits = probs.copy()
            d_logits[range(curr_bs), yb] -= 1
            d_logits /= curr_bs
            
            model.backward(d_logits)
            
            epoch_loss += loss * (curr_bs / N)
            epoch_acc += acc * (curr_bs / N)
            
        if epoch % 5 == 0:
            elapsed = time.time() - start_time
            print(f"EVO_EPOCH:{epoch:03d} | LOSS:{epoch_loss:.4f} | ACC:{epoch_acc:.4f} | TIME:{elapsed:.2f}s")

    print("PHASE: EVOLUTION_COMPLETE | STATUS: SUPREME")

if __name__ == "__main__":
    train_evolution()
