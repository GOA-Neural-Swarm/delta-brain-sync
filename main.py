
import numpy as np
import time

class Linear:
    def __init__(self, in_d, out_d, use_bias=True):
        limit = np.sqrt(2.0 / (in_d + out_d))
        self.W = np.random.randn(in_d, out_d).astype(np.float32) * limit
        self.b = np.zeros(out_d, dtype=np.float32) if use_bias else None
        self.x, self.dW, self.db = None, None, None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + (self.b if self.b is not None else 0)

    def backward(self, dout):
        self.dW = np.dot(self.x.T, dout)
        if self.b is not None:
            self.db = np.sum(dout, axis=0)
        return np.dot(dout, self.W.T)

    def params(self):
        p = [{"ref": self.W, "grad": self.dW}]
        if self.b is not None:
            p.append({"ref": self.b, "grad": self.db})
        return p


class RMSNorm:
    def __init__(self, dim, eps=1e-6):
        self.scale = np.ones(dim, dtype=np.float32)
        self.eps = eps
        self.x, self.rstd, self.dscale = None, None, None

    def forward(self, x):
        self.x = x
        var = np.mean(x**2, axis=-1, keepdims=True)
        self.rstd = 1.0 / np.sqrt(var + self.eps)
        return self.scale * (x * self.rstd)

    def backward(self, dout):
        x_rstd = self.x * self.rstd
        self.dscale = np.sum(dout * x_rstd, axis=0)
        dx_rstd = dout * self.scale
        m_dx_rstd_x_rstd = np.mean(dx_rstd * x_rstd, axis=-1, keepdims=True)
        return self.rstd * (dx_rstd - x_rstd * m_dx_rstd_x_rstd)

    def params(self):
        return [{"ref": self.scale, "grad": self.dscale}]


class SwiGLU:
    def __init__(self, dim, h_dim):
        self.w1 = Linear(dim, h_dim, use_bias=False)
        self.w2 = Linear(dim, h_dim, use_bias=False)
        self.w3 = Linear(h_dim, dim, use_bias=False)
        self.x1, self.x2, self.sig, self.swish = None, None, None, None

    def forward(self, x):
        self.x1 = self.w1.forward(x)
        self.x2 = self.w2.forward(x)
        self.sig = 1.0 / (1.0 + np.exp(-np.clip(self.x1, -10, 10)))
        self.swish = self.x1 * self.sig
        return self.w3.forward(self.swish * self.x2)

    def backward(self, dout):
        dw3 = self.w3.backward(dout)
        dx2 = dw3 * self.swish
        dswish = dw3 * self.x2
        dx1 = dswish * (self.sig * (1.0 + self.x1 * (1.0 - self.sig)))
        return self.w1.backward(dx1) + self.w2.backward(dx2)

    def params(self):
        return self.w1.params() + self.w2.params() + self.w3.params()


class RedundantMoE:
    def __init__(self, dim):
        self.gemini_engine = SwiGLU(dim, dim * 2)
        self.groq_engine = SwiGLU(dim, dim * 2)
        self.gate = Linear(dim, 2, use_bias=False)
        self.probs, self.out_gemini, self.out_groq = None, None, None

    def forward(self, x):
        logits = self.gate.forward(x)
        logits -= np.max(logits, axis=-1, keepdims=True)
        exp_l = np.exp(logits)
        self.probs = exp_l / (np.sum(exp_l, axis=-1, keepdims=True) + 1e-10)
        self.out_gemini = self.gemini_engine.forward(x)
        self.out_groq = self.groq_engine.forward(x)
        return self.probs[:, 0:1] * self.out_gemini + self.probs[:, 1:2] * self.out_groq

    def backward(self, dout):
        p0, p1 = self.probs[:, 0:1], self.probs[:, 1:2]
        d_gemini = self.gemini_engine.backward(dout * p0)
        d_groq = self.groq_engine.backward(dout * p1)
        dp0 = np.sum(dout * self.out_gemini, axis=-1, keepdims=True)
        dp1 = np.sum(dout * self.out_groq, axis=-1, keepdims=True)
        d_logits_raw = np.concatenate([dp0, dp1], axis=-1)
        d_gate_logits = self.probs * (
            d_logits_raw - np.sum(self.probs * d_logits_raw, axis=-1, keepdims=True)
        )
        dx_gate = self.gate.backward(d_gate_logits)
        return d_gemini + d_groq + dx_gate

    def params(self):
        return (
            self.gemini_engine.params()
            + self.groq_engine.params()
            + self.gate.params()
        )


class SovereignBlock:
    def __init__(self, dim):
        self.norm1 = RMSNorm(dim)
        self.moe = RedundantMoE(dim)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, dim * 4)
        self.alpha = np.array([0.5], dtype=np.float32)
        self.beta = np.array([0.5], dtype=np.float32)
        self.dalpha, self.dbeta = None, None
        self.moe_res, self.mlp_res = None, None

    def forward(self, x):
        self.moe_res = self.moe.forward(self.norm1.forward(x))
        x = x + self.alpha * self.moe_res
        self.mlp_res = self.mlp.forward(self.norm2.forward(x))
        return x + self.beta * self.mlp_res

    def backward(self, dout):
        self.dbeta = np.sum(dout * self.mlp_res, keepdims=True).flatten()
        dmlp = self.mlp.backward(dout * self.beta)
        dn2 = self.norm2.backward(dmlp)
        dx_mid = dout + dn2
        self.dalpha = np.sum(dx_mid * self.moe_res, keepdims=True).flatten()
        dmoe = self.moe.backward(self.norm1.backward(dx_mid * self.alpha))
        return dx_mid + dmoe

    def params(self):
        p = (
            self.norm1.params()
            + self.moe.params()
            + self.norm2.params()
            + self.mlp.params()
        )
        p.extend(
            [
                {"ref": self.alpha, "grad": self.dalpha},
                {"ref": self.beta, "grad": self.dbeta},
            ]
        )
        return p


class SovereignArchitect:
    def __init__(self, in_d, h_d, out_d, depth):
        self.stem = Linear(in_d, h_d)
        self.blocks = [SovereignBlock(h_d) for _ in range(depth)]
        self.norm_f = RMSNorm(h_d)
        self.head = Linear(h_d, out_d)

    def forward(self, x):
        x = self.stem.forward(x)
        for b in self.blocks:
            x = b.forward(x)
        return self.head.forward(self.norm_f.forward(x))

    def backward(self, dout):
        dout = self.norm_f.backward(self.head.backward(dout))
        for b in reversed(self.blocks):
            dout = b.backward(dout)
        self.stem.backward(dout)

    def params(self):
        p = self.stem.params()
        for b in self.blocks:
            p.extend(b.params())
        p.extend(self.norm_f.params())
        p.extend(self.head.params())
        return p


class Lion:
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), wd=0.01):
        self.params = params
        self.lr, self.beta1, self.beta2, self.wd = lr, betas[0], betas[1], wd
        self.m = [np.zeros_like(p["ref"]) for p in params]

    def step(self, lr_mult=1.0):
        curr_lr = self.lr * lr_mult
        for i, p in enumerate(self.params):
            param, grad = p["ref"], p["grad"]
            if grad is None:
                continue

            if self.wd > 0:
                param -= curr_lr * self.wd * param

            update = np.sign(self.beta1 * self.m[i] + (1 - self.beta1) * grad)
            param -= curr_lr * update

            self.m[i] = self.beta2 * self.m[i] + (1 - self.beta2) * grad


def evolve():
    N, D, K = 10000, 784, 10
    X = np.random.randn(N, D).astype(np.float32)
    centers = np.random.randn(K, D).astype(np.float32) * 2.0
    y = np.random.randint(0, K, N)
    X += centers[y]

    model = SovereignArchitect(D, 256, K, depth=6)
    optimizer = Lion(model.params(), lr=1e-4, wd=0.02)

    batch_size, epochs = 256, 50
    print("OMEGA-ASI | RECURSIVE EVOLUTION | ARCHITECTURE: SOVEREIGN-V8-HYPER")

    for epoch in range(epochs):
        idx = np.random.permutation(N)
        total_loss, total_acc = 0, 0
        t0 = time.time()

        lr_mult = 0.5 * (1 + np.cos(np.pi * epoch / epochs))
        if epoch < 5:
            lr_mult *= (epoch + 1) / 5

        for i in range(0, N, batch_size):
            batch_idx = idx[i : i + batch_size]
            xb, yb = X[batch_idx], y[batch_idx]
            m = xb.shape[0]

            logits = model.forward(xb)
            logits -= np.max(logits, axis=1, keepdims=True)
            exp_l = np.exp(logits)
            probs = exp_l / (np.sum(exp_l, axis=1, keepdims=True) + 1e-10)

            loss = -np.mean(np.log(probs[range(m), yb] + 1e-10))
            total_loss += loss * (m / N)
            total_acc += np.mean(np.argmax(probs, axis=1) == yb) * (m / N)

            dout = probs.copy()
            dout[range(m), yb] -= 1
            model.backward(dout / m)

            gnorm = np.sqrt(
                sum(
                    np.sum(p["grad"] ** 2)
                    for p in model.params()
                    if p["grad"] is not None
                )
            )
            if gnorm > 1.0:
                for p in model.params():
                    if p["grad"] is not None:
                        p["grad"] *= 1.0 / (gnorm + 1e-6)

            optimizer.step(lr_mult=lr_mult)

        dt = time.time() - t0
        print(
            f"EP:{epoch:02d} | LOSS:{total_loss:.4f} | ACC:{total_acc:.4f} | SPEED:{N/dt:.0f}sps | LR:{optimizer.lr*lr_mult:.7f}"
        )


if __name__ == "__main__":
    evolve()
