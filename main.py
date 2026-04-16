import numpy as np

def silu(x): return x / (1 + np.exp(-np.clip(x, -12, 12)))

class L:
    def __init__(self, i, o, s=None):
        self.W, self.b = np.random.randn(i, o).astype("f4") * (s or (2/i)**.5), np.zeros(o, "f4")
    def forward(self, x):
        self.x = x
        return x @ self.W + self.b
    def backward(self, dy):
        x_f, dy_f = self.x.reshape(-1, self.x.shape[-1]), dy.reshape(-1, dy.shape[-1])
        self.dW, self.db = x_f.T @ dy_f, dy_f.sum(0)
        return (dy @ self.W.T).reshape(self.x.shape[:-1] + (self.W.shape[0],))

class RN:
    def __init__(self, d): self.g = np.ones(d, "f4")
    def forward(self, x):
        self.x, self.v = x, np.mean(x**2, -1, keepdims=True)
        self.r = (self.v + 1e-6)**-0.5
        return self.g * (x * self.r)
    def backward(self, dy):
        xn, dn = self.x * self.r, dy * self.g
        self.dg = (dy * xn).reshape(-1, xn.shape[-1]).sum(0)
        return self.r * (dn - xn * np.mean(dn * xn, -1, keepdims=True))

class RP:
    def __init__(self, d, m=4096):
        f = 10000**-(np.arange(0, d, 2)/d)
        t = np.outer(np.arange(m), f)
        self.c, self.s = np.cos(t), np.sin(t)
    def apply(self, x, conj=False):
        s, d = x.shape[1], x.shape[-1]
        x1, x2 = x[..., :d//2], x[..., d//2:]
        c, sn = self.c[:s, None, :], self.s[:s, None, :]
        return np.concatenate([x1*c+x2*sn, x2*c-x1*sn] if conj else [x1*c-x2*sn, x2*c+x1*sn], -1)

class GQA:
    def __init__(self, d, h=8, g=2):
        self.d, self.h, self.g, self.hd = d, h, g, d // h
        self.wq, self.wk, self.wv, self.wo = L(d, d), L(d, (h//g)*self.hd), L(d, (h//g)*self.hd), L(d, d)
        self.rp, self.sc = RP(self.hd), (d//h)**-0.5
    def forward(self, x):
        b, s = x.shape[:2]
        q, k, v = self.wq.forward(x).reshape(b,s,self.h,self.hd), self.wk.forward(x).reshape(b,s,self.h//self.g,self.hd), self.wv.forward(x).reshape(b,s,self.h//self.g,self.hd)
        self.qr, self.kr, self.v_raw = self.rp.apply(q), self.rp.apply(k), v
        ke, ve = np.repeat(self.kr, self.g, 2), np.repeat(v, self.g, 2)
        at = np.einsum("bshd,bthd->bsht", self.qr, ke) * self.sc
        at -= at.max(-1, keepdims=True)
        self.p = np.exp(at); self.p /= self.p.sum(-1, keepdims=True) + 1e-12
        return self.wo.forward(np.einsum("bsht,bthd->bshd", self.p, ve).reshape(b, s, -1))
    def backward(self, dy):
        b, s = dy.shape[:2]
        dy_o = self.wo.backward(dy).reshape(b, s, self.h, self.hd)
        ke, ve = np.repeat(self.kr, self.g, 2), np.repeat(self.v_raw, self.g, 2)
        dp = np.einsum("bshd,bthd->bsht", dy_o, ve)
        da = self.p * (dp - (self.p * dp).sum(-1, keepdims=True)) * self.sc
        dq = self.rp.apply(np.einsum("bsht,bthd->bshd", da, ke), True)
        dk = self.rp.apply(np.einsum("bsht,bshd->bthd", da, self.qr).reshape(b,s,self.h//self.g,self.g,self.hd).sum(3), True)
        dv = np.einsum("bsht,bshd->bthd", self.p, dy_o).reshape(b,s,self.h//self.g,self.g,self.hd).sum(3)
        return self.wq.backward(dq.reshape(b,s,-1)) + self.wk.backward(dk.reshape(b,s,-1)) + self.wv.backward(dv.reshape(b,s,-1))

class SG:
    def __init__(self, d, h): self.w1, self.w2, self.w3 = L(d, h), L(h, d), L(d, h)
    def forward(self, x):
        self.x1, self.x3 = self.w1.forward(x), self.w3.forward(x)
        self.sig = 1 / (1 + np.exp(-np.clip(self.x1, -12, 12)))
        return self.w2.forward((self.x1 * self.sig) * self.x3)
    def backward(self, dy):
        da = self.w2.backward(dy)
        dx1 = da * self.x3 * (self.sig * (1 + self.x1 * (1 - self.sig)))
        return self.w1.backward(dx1) + self.w3.backward(da * (self.x1 * self.sig))

class MoE:
    def __init__(self, d, n=4, e=2):
        self.n, self.gate, self.ex = n, L(d, n), [SG(d, d*e) for _ in range(n)]
    def forward(self, x):
        self.x = x
        g = self.gate.forward(x)
        self.pr = np.exp(g - g.max(-1, keepdims=True)); self.pr /= self.pr.sum(-1, keepdims=True)
        self.eo = [exp.forward(x) for exp in self.ex]
        return sum(self.pr[..., i:i+1] * self.eo[i] for i in range(self.n))
    def backward(self, dy):
        dx, dpr = np.zeros_like(self.x), np.zeros_like(self.pr)
        for i in range(self.n):
            dpr[..., i] = (dy * self.eo[i]).sum(-1)
            dx += self.ex[i].backward(dy * self.pr[..., i:i+1])
        dg = self.pr * (dpr - (self.pr * dpr).sum(-1, keepdims=True))
        return dx + self.gate.backward(dg)

class RB:
    def __init__(self, d):
        self.n1, self.at, self.n2, self.mo, self.a = RN(d), GQA(d), RN(d), MoE(d), np.array([0.5], "f4")
    def forward(self, x):
        self.x, self.ga, self.qa = x, self.at.forward(self.n1.forward(x)), self.mo.forward(self.n2.forward(x))
        return x + self.a * self.ga + (1 - self.a) * self.qa
    def backward(self, dy):
        self.da = (dy * (self.ga - self.qa)).sum()
        return dy + self.n1.backward(self.at.backward(dy * self.a)) + self.n2.backward(self.mo.backward(dy * (1 - self.a)))

class ASI:
    def __init__(self, i, h, o, d=2):
        self.emb, self.blks, self.norm, self.head = L(i, h), [RB(h) for _ in range(d)], RN(h), L(h, o)
    def forward(self, x):
        x = self.emb.forward(x[:, None] if x.ndim==2 else x)
        for b in self.blks: x = b.forward(x)
        return self.head.forward(self.norm.forward(x[:, -1]))
    def backward(self, dy):
        dy = self.norm.backward(self.head.backward(dy))
        dys = np.zeros((dy.shape[0], 1, dy.shape[1]), "f4")
        dys[:, -1] = dy
        for b in reversed(self.blks): dys = b.backward(dys)
        self.emb.backward(dys)
    def params(self):
        res = []
        def col(o):
            if isinstance(o, (L, RN)): res.append(o)
            elif isinstance(o, list): [col(i) for i in o]
            elif hasattr(o, "__dict__"): [col(v) for k, v in o.__dict__.items() if k[0] != "_"]
        col(self); return list(set(res))

class Opt:
    def __init__(self, p, lr=1e-3):
        self.p, self.lr, self.t = p, lr, 0
        self.m = {id(i): [np.zeros_like(getattr(i, a)) for a in (["W","b"] if hasattr(i,"W") else ["g"])] for i in p}
        self.v = {id(i): [np.zeros_like(getattr(i, a)) for a in (["W","b"] if hasattr(i,"W") else ["g"])] for i in p}
    def step(self):
        self.t += 1
        lr = self.lr * min(1.0, self.t / 50) * (0.5 * (1 + np.cos(min(self.t, 500) * np.pi / 500)))
        for p in self.p:
            at = ["W", "b"] if hasattr(p, "W") else ["g"]
            for i, a in enumerate(at):
                g = getattr(p, "d"+a if a!="g" else "dg")
                self.m[id(p)][i] = 0.9 * self.m[id(p)][i] + 0.1 * g
                self.v[id(p)][i] = 0.99 * self.v[id(p)][i] + 0.01 * (g**2)
                mh, vh = self.m[id(p)][i]/(1-0.9**self.t), self.v[id(p)][i]/(1-0.99**self.t)
                setattr(p, a, getattr(p, a) - lr * (mh / (np.sqrt(vh) + 1e-8) + 0.01 * getattr(p, a)))

def train():
    N, D, C, BS, E = 1024, 784, 10, 32, 50
    X, Y = np.random.randn(N, D).astype("f4"), np.random.randint(0, C, N)
    m = ASI(D, 128, C, 2); ps = m.params(); opt = Opt(ps, 2e-3)
    for e in range(E):
        idx, loss, acc = np.random.permutation(N), 0, 0
        for i in range(0, N, BS):
            xb, yb = X[idx[i:i+BS]], Y[idx[i:i+BS]]
            lg = m.forward(xb)
            p = np.exp(lg - lg.max(-1, 1)); p /= p.sum(-1, 1)
            loss += -np.log(p[range(len(yb)), yb] + 1e-12).sum()
            acc += (p.argmax(-1) == yb).sum()
            dy = p.copy(); dy[range(len(yb)), yb] -= 1
            m.backward(dy / len(yb))
            gn = np.sqrt(sum((getattr(p, "d"+a if a!="g" else "dg")**2).sum() for p in ps for a in (["W","b"] if hasattr(p,"W") else ["g"])))
            if gn > 1.0:
                for p in ps:
                    for a in (["W","b"] if hasattr(p,"W") else ["g"]):
                        k = "d"+a if a!="g" else "dg"
                        setattr(p, k, getattr(p, k)/gn)
            opt.step()
        if (e+1) % 5 == 0: print(f"E {e+1:02d} | L: {loss/N:.4f} | A: {acc/N:.4f}")

if __name__ == "__main__": train()