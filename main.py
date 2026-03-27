import numpy as np
import time

class Tensor:
    def __init__(self, data, name=""):
        self.d = data.astype('float32')
        self.g = np.zeros_like(self.d)
        self.m = np.zeros_like(self.d)
        self.v = np.zeros_like(self.d)

class Module:
    def params(self):
        p = []
        for v in self.__dict__.values():
            if isinstance(v, Tensor): p.append(v)
            elif isinstance(v, Module): p.extend(v.params())
            elif isinstance(v, list):
                for i in v:
                    if isinstance(i, Module): p.extend(i.params())
        return p

class Linear(Module):
    def __init__(self, i, o, bias=True):
        self.w = Tensor(np.random.randn(i, o) * np.sqrt(2.0 / i))
        self.b = Tensor(np.zeros((1, o))) if bias else None

    def f(self, x):
        self.x = x
        return x @ self.w.d + (self.b.d if self.b else 0)

    def b(self, g):
        self.w.g += self.x.T @ g
        if self.b: self.b.g += np.sum(g, axis=0, keepdims=True)
        return g @ self.w.d.T

class RMSNorm(Module):
    def __init__(self, d, e=1e-6):
        self.g, self.e = Tensor(np.ones((1, d))), e

    def f(self, x):
        self.x = x
        self.ms = np.mean(x**2, axis=-1, keepdims=True)
        self.rms = np.sqrt(self.ms + self.e)
        self.xh = x / self.rms
        return self.g.d * self.xh

    def b(self, g):
        self.g.g += np.sum(g * self.xh, axis=0, keepdims=True)
        dxh = g * self.g.d
        n = self.x.shape[-1]
        return (1.0 / self.rms) * (dxh - self.xh * np.mean(dxh * self.xh, axis=-1, keepdims=True))

class SwiGLU(Module):
    def __init__(self, d, h):
        self.w1 = Linear(d, h, False)
        self.w2 = Linear(d, h, False)
        self.w3 = Linear(h, d, False)

    def f(self, x):
        self.x1 = self.w1.f(x)
        self.x2 = self.w2.f(x)
        self.sig = 1.0 / (1.0 + np.exp(-np.clip(self.x1, -10, 10)))
        self.swish = self.x1 * self.sig
        self.out = self.swish * self.x2
        return self.w3.f(self.out)

    def b(self, g):
        g3 = self.w3.b(g)
        dx2 = g3 * self.swish
        dswish = g3 * self.x2
        dx1 = dswish * (self.sig * (1.0 + self.x1 * (1.0 - self.sig)))
        return self.w1.b(dx1) + self.w2.b(dx2)

class RedundantEngine(Module):
    def __init__(self, d, h):
        self.groq_path = Linear(d, h, False)
        self.gemini_path = SwiGLU(d, h)
        self.gate = Tensor(np.zeros((1, d)))
        self.proj = Linear(h, d, False)

    def f(self, x):
        self.x = x
        self.g_val = 1.0 / (1.0 + np.exp(-self.gate.d))
        self.o_groq = self.groq_path.f(x)
        self.o_gemini = self.gemini_path.f(x)
        # Groq path is high-throughput linear, Gemini is deep SwiGLU
        # Redundancy logic: learnable fusion of deterministic and stochastic paths
        return x + self.g_val * self.proj.f(self.o_groq) + (1.0 - self.g_val) * self.o_gemini

    def b(self, g):
        dg_gate = np.sum(g * (self.proj.f(self.o_groq) - self.o_gemini) * (self.g_val * (1.0 - self.g_val)), axis=0, keepdims=True)
        self.gate.g += dg_gate
        g_groq = self.proj.b(g * self.g_val)
        dx_groq = self.groq_path.b(g_groq)
        dx_gemini = self.gemini_path.b(g * (1.0 - self.g_val))
        return g + dx_groq + dx_gemini

class EvolutionBlock(Module):
    def __init__(self, d, h):
        self.ln1 = RMSNorm(d)
        self.engine = RedundantEngine(d, h)
        self.ln2 = RMSNorm(d)
        self.mlp = SwiGLU(d, h)

    def f(self, x):
        x = x + self.engine.f(self.ln1.f(x))
        x = x + self.mlp.f(self.ln2.f(x))
        return x

    def b(self, g):
        g_mlp = self.mlp.b(g)
        g = g + self.ln2.b(g_mlp)
        g_eng = self.engine.b(g)
        g = g + self.ln1.b(g_eng)
        return g

class AdamW:
    def __init__(self, p, lr=1e-3, b=(0.9, 0.95), e=1e-8, wd=0.01):
        self.p, self.lr, self.b, self.e, self.wd, self.t = p, lr, b, e, wd, 0

    def step(self):
        self.t += 1
        at = self.lr * np.sqrt(1.0 - self.b[1]**self.t) / (1.0 - self.b[0]**self.t)
        for p in self.p:
            p.d -= self.lr * self.wd * p.d
            p.m = self.b[0] * p.m + (1.0 - self.b[0]) * p.g
            p.v = self.b[1] * p.v + (1.0 - self.b[1]) * (p.g**2)
            p.d -= at * p.m / (np.sqrt(p.v) + self.e)

class OMEGA_ASI(Module):
    def __init__(self, i, h, o, d=4):
        self.emb = Linear(i, h)
        self.blocks = [EvolutionBlock(h, h * 2) for _ in range(d)]
        self.norm = RMSNorm(h)
        self.head = Linear(h, o)
        self.params_list = self.params()
        self.opt = AdamW(self.params_list, lr=1e-3, wd=0.05)

    def f(self, x):
        x = self.emb.f(x)
        for b in self.blocks: x = b.f(x)
        return self.head.f(self.norm.f(x))

    def b(self, g):
        g = self.norm.b(self.head.b(g))
        for b in reversed(self.blocks): g = b.b(g)
        self.emb.b(g)

    def step(self, x, y):
        for p in self.params_list: p.g.fill(0)
        logits = self.f(x)
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exps / (np.sum(exps, axis=1, keepdims=True) + 1e-10)
        loss = -np.mean(np.sum(y * np.log(probs + 1e-10), axis=1))
        self.b((probs - y) / x.shape[0])
        gn = np.sqrt(sum(np.sum(p.g**2) for p in self.params_list))
        if gn > 1.0:
            for p in self.params_list: p.g /= gn
        self.opt.step()
        return loss

def generate_synthetic_data(n=10000, d=784, c=10):
    X = np.random.randn(n, d).astype('float32')
    # Complex non-linear relationship
    W1 = np.random.randn(d, 512)
    W2 = np.random.randn(512, c)
    Z = np.maximum(0, X @ W1)
    Y_logits = Z @ W2
    Y_idx = np.argmax(Y_logits, axis=1)
    X = (X - np.mean(X)) / (np.std(X) + 1e-7)
    return X, np.eye(c)[Y_idx].astype('float32')

if __name__ == "__main__":
    print("SYSTEM: OMEGA-ASI | CORE: RECURSIVE-EVOLUTION | MODE: HIGH-PERFORMANCE")
    X, Y = generate_synthetic_data(15000)
    model = OMEGA_ASI(784, 128, 10, 4)
    batch_size = 64
    epochs = 30
    lr_max = 3e-3

    for epoch in range(1, epochs + 1):
        model.opt.lr = lr_max * 0.5 * (1 + np.cos(np.pi * epoch / epochs))
        indices = np.random.permutation(len(X))
        losses = []
        start_time = time.time()
        
        for i in range(0, len(X), batch_size):
            batch_idx = indices[i:i+batch_size]
            loss = model.step(X[batch_idx], Y[batch_idx])
            losses.append(loss)
            
        # Validation
        v_idx = np.random.choice(len(X), 500)
        v_logits = model.f(X[v_idx])
        acc = np.mean(np.argmax(v_logits, 1) == np.argmax(Y[v_idx], 1))
        
        print(f"EPOCH: {epoch:02d} | LOSS: {np.mean(losses):.5f} | ACC: {acc:.4f} | LR: {model.opt.lr:.6f} | T: {time.time()-start_time:.2f}s")
        if acc > 0.995: 
            print("EVOLUTIONARY TARGET REACHED.")
            break
