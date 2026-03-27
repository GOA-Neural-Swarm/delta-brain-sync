import numpy as np
import time

class P:
    def __init__(s, d, n=""):
        s.d = d.astype('f4')
        s.g = s.m = s.v = np.zeros_like(s.d)

class M:
    def __init__(s): s.t = 1
    def p(s):
        r = []
        for v in s.__dict__.values():
            if isinstance(v, P): r += [v]
            elif isinstance(v, M): r += v.p()
            elif isinstance(v, list): [r.extend(i.p()) for i in v if isinstance(i, M)]
        return r

class L(M):
    def __init__(s, i, o, b=1):
        super().__init__()
        s.w = P(np.random.randn(i, o) * (2/i)**.5)
        s.b = P(np.zeros((1, o))) if b else None
    def f(s, x):
        s.x = x
        o = x @ s.w.d
        return o + s.b.d if s.b else o
    def b(s, g):
        s.w.g = s.x.T @ g
        if s.b: s.b.g = g.sum(0, keepdims=1)
        return g @ s.w.d.T

class R(M):
    def __init__(s, d, e=1e-6):
        super().__init__()
        s.g, s.e = P(np.ones((1, d))), e
    def f(s, x):
        s.x = x
        s.rms = np.sqrt((x**2).mean(-1, keepdims=1) + s.e)
        s.xh = x / s.rms
        return s.g.d * s.xh
    def b(s, g):
        s.g.g = (g * s.xh).sum(0, keepdims=1)
        dxh = g * s.g.d
        return (dxh - s.xh * (dxh * s.xh).mean(-1, keepdims=1)) / s.rms

class S(M):
    def __init__(s, d, h):
        super().__init__()
        s.w1, s.w2, s.w3 = L(d, h, 0), L(d, h, 0), L(h, d, 0)
    def f(s, x):
        s.x1, s.x2 = s.w1.f(x), s.w2.f(x)
        s.sig = 1 / (1 + np.exp(-s.x1))
        s.sw = s.x1 * s.sig
        return s.w3.f(s.sw * s.x2)
    def b(s, g):
        g = s.w3.b(g)
        dx2, dsw = g * s.sw, g * s.x2
        dx1 = dsw * (s.sig * (1 + s.x1 * (1 - s.sig)))
        return s.w1.b(dx1) + s.w2.b(dx2)

class G(M):
    def f(s, x):
        s.x = x
        s.sig = 1 / (1 + np.exp(-1.702 * x))
        return x * s.sig
    def b(s, g):
        return g * (s.sig + 1.702 * s.x * s.sig * (1 - s.sig))

class B(M):
    def __init__(s, d):
        super().__init__()
        s.n1, s.n2 = R(d), R(d)
        s.gp, s.p1, s.ac, s.p2 = S(d, d*4), L(d, d*4), G(), L(d*4, d)
        s.gt = P(np.zeros((1, d)))
    def f(s, x):
        s.r = x
        nx = s.n1.f(x)
        s.oa, s.ob = s.gp.f(nx), s.p2.f(s.ac.f(s.p1.f(nx)))
        s.gv = 1 / (1 + np.exp(-s.gt.d))
        return s.r + s.gv * s.oa + (1 - s.gv) * s.ob
    def b(s, g):
        dg = s.gv * (1 - s.gv)
        s.gt.g = (g * (s.oa - s.ob) * dg).sum(0, keepdims=1)
        ga = s.gp.b(g * s.gv)
        gb = s.p1.b(s.ac.b(s.p2.b(g * (1 - s.gv))))
        return s.n1.b(ga + gb) + g

class A:
    def __init__(s, p, lr=1e-3, b=(.9, .999), e=1e-8, w=.01):
        s.p, s.lr, s.b, s.e, s.w, s.t = p, lr, b, e, w, 0
    def step(s):
        s.t += 1
        a = s.lr * (1 - s.b[1]**s.t)**.5 / (1 - s.b[0]**s.t)
        for p in s.p:
            if s.w > 0: p.d -= s.lr * s.w * p.d
            p.m = s.b[0] * p.m + (1 - s.b[0]) * p.g
            p.v = s.b[1] * p.v + (1 - s.b[1]) * (p.g**2)
            p.d -= a * p.m / (np.sqrt(p.v) + s.e)

class O(M):
    def __init__(s, i, h, o, d=4):
        super().__init__()
        s.st, s.bl = L(i, h), [B(h) for _ in range(d)]
        s.hn, s.hd = R(h), L(h, o)
        s.ps = s.p()
        s.opt = A(s.ps, lr=1e-3, w=.05)
    def f(s, x):
        x = s.st.f(x)
        for b in s.bl: x = b.f(x)
        return s.hd.f(s.hn.f(x))
    def b(s, g):
        g = s.hn.b(s.hd.b(g))
        for b in reversed(s.bl): g = b.b(g)
        s.st.b(g)
    def step(s, x, y):
        lgt = s.f(x)
        ex = np.exp(lgt - lgt.max(1, keepdims=1))
        pr = ex / ex.sum(1, keepdims=1)
        loss = -np.mean((y * np.log(pr + 1e-12)).sum(1))
        s.b((pr - y) / y.shape[0])
        gn = np.sqrt(sum((p.g**2).sum() for p in s.ps))
        if gn > 1:
            for p in s.ps: p.g /= gn
        s.opt.step()
        return loss

def get_data(n=5000, d=784, c=10):
    x = np.random.randn(n, d).astype('f4')
    y_idx = np.argmax(x @ np.random.randn(d, c) + .05 * np.random.randn(n, c), 1)
    return (x - x.mean()) / x.std(), np.eye(c)[y_idx].astype('f4')

if __name__ == "__main__":
    X, Y = get_data(10000)
    m = O(784, 128, 10, 4)
    bs, ep = 64, 50
    for e in range(1, ep + 1):
        idx = np.random.permutation(len(X))
        ls, t0 = [], time.time()
        for i in range(0, len(X), bs):
            ls.append(m.step(X[idx[i:i+bs]], Y[idx[i:i+bs]]))
        v_l = m.f(X[:500])
        acc = (v_l.argmax(1) == Y[:500].argmax(1)).mean()
        print(f"C {e:02} | L: {np.mean(ls):.4f} | A: {acc:.4f} | T: {time.time()-t0:.1f}s")
        if acc > .99: break