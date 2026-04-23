import numpy as np


class Tensor:
    def __init__(self, d):
        self.d = d.astype("f4")
        self.g = np.zeros_like(d, dtype="f4")


class Linear:
    def __init__(self, i, o):
        s = np.sqrt(2.0 / (i + o))
        self.w = Tensor(np.random.normal(0, s, (i, o)))
        self.b = Tensor(np.zeros(o))

    def forward(self, x):
        self.x = x
        return x @ self.w.d + self.b.d
    def b(self, dy):
        dx = dy.reshape(-1, dy.shape[-1])
        self.w.g += self.x.reshape(-1, self.x.shape[-1]).T @ dx
        self.b.g += dx.sum(0)
        return dy @ self.w.d.T


class RMSNorm:
    def __init__(self, d, e=1e-6):
        self.g, self.e = Tensor(np.ones(d)), e

    def forward(self, x):
        self.x = x
        self.v = np.mean(x**2, -1, keepdims=True)
        self.isd = 1.0 / np.sqrt(self.v + self.e)
        self.nx = x * self.isd
        return self.g.d * self.nx
    def b(self, dy):
        self.g.g += np.sum(dy * self.nx, axis=tuple(range(dy.ndim - 1)))
        dn = dy * self.g.d
        return (dn - self.nx * np.mean(dn * self.nx, -1, keepdims=True)) * self.isd


class SwiGLU:
    def forward(self, x):
        x, g = np.split(x, 2, -1)
        self.s = 1.0 / (1.0 + np.exp(-np.clip(g, -15, 15)))
        self.sw, self.x, self.g = g * self.s, x, g
        return x * self.sw
    def b(self, dy):
        dx = dy * self.sw
        dg = dy * self.x * self.s * (1.0 + self.g * (1.0 - self.s))
        return np.concatenate([dx, dg], -1)


class GQA:
    def __init__(self, d, h=8, g=2):
        self.d, self.h, self.g, self.hd = d, h, g, d // h
        self.wq, self.wk, self.wv, self.wo = (
            Linear(d, d),
            Linear(d, (h // g) * self.hd),
            Linear(d, (h // g) * self.hd),
            Linear(d, d),
        )
        self.scale = self.hd**-0.5

    def forward(self, x):
        b, s, _ = x.shape
        q = self.wq.forward(x).reshape(b, s, self.h, self.hd)
        k = self.wk.forward(x).reshape(b, s, self.h // self.g, self.hd)
        v = self.wv.forward(x).reshape(b, s, self.h // self.g, self.hd)

        p, f = np.arange(s)[:, None], 10000 ** -(np.arange(0, self.hd, 2) / self.hd)
        a = p * f
        co, si = np.cos(a), np.sin(a)

        def rope(t):
            r, i = t[..., ::2], t[..., 1::2]
            return np.stack(
                [
                    r * co[:, None, :] - i * si[:, None, :],
                    r * si[:, None, :] + i * co[:, None, :],
                ],
                -1,
            ).reshape(t.shape)

        self.qr, self.kr, self.vr = rope(q), rope(k), v
        kr, vr = np.repeat(self.qr, self.g, 2), np.repeat(v, self.g, 2)
        at = (
            np.einsum("bshd,bthd->bsht", self.qr, np.repeat(self.kr, self.g, 2))
            * self.scale
        )
        at = np.exp(at - np.max(at, -1, keepdims=True))
        self.p = at / (at.sum(-1, keepdims=True) + 1e-12)
        out = np.einsum("bsht,bthd->bshd", self.p, np.repeat(v, self.g, 2)).reshape(b, s, -1)
        return self.wo.f(out)
    def b(self, dy):
        b, s = dy.shape[:2]
        dw = self.wo.b(dy).reshape(b, s, self.h, self.hd)
        vr_r = np.repeat(self.vr, self.g, 2)
        dp = np.einsum("bshd,bthd->bsht", dw, vr_r)
        da = self.p * (dp - np.sum(self.p * dp, -1, keepdims=True)) * self.sc
        dqr = np.einsum("bsht,bthd->bshd", da, np.repeat(self.kr, self.g, 2))
        dkr = (
            np.einsum("bsht,bshd->bthd", da, self.qr)
            .reshape(b, s, self.h // self.g, self.g, self.hd)
            .sum(3)
        )
        dv = (
            np.einsum("bsht,bshd->bthd", self.p, dw)
            .reshape(b, s, self.h // self.g, self.g, self.hd)
            .sum(3)
        )
        return (
            self.wq.backward(dqr.reshape(b, s, -1))
            + self.wk.backward(dkr.reshape(b, s, -1))
            + self.wv.backward(dv.reshape(b, s, -1))
        )


class MOE:
    def __init__(self, d, n=4, k=2):
        self.d, self.n, self.k = d, n, k
        self.gate = Linear(d, n)
        self.w1 = [
            Tensor(np.random.normal(0, np.sqrt(2 / (d + d * 2)), (d, d * 2)))
            for _ in range(n)
        ]
        self.w2 = [
            Tensor(np.random.normal(0, np.sqrt(1 / d), (d * 2, d))) for _ in range(n)
        ]
        self.swi = [SwiGLU() for _ in range(n)]

    def forward(self, x):
        os = x.shape
        x = x.reshape(-1, self.d)
        lg = self.gate.f(x)
        pr = np.exp(lg - lg.max(-1, keepdims=True))
        pr /= pr.sum(-1, keepdims=True)
        self.idx = np.argsort(pr, -1)[:, -self.k :]
        self.w = np.take_along_axis(pr, self.idx, -1)
        self.w /= self.w.sum(-1, keepdims=True) + 1e-12
        out = np.zeros_like(x)
        self.cache = []
        for i in range(self.n):
            m = np.any(self.idx == i, -1)
            if not np.any(m):
                self.cache.append(None)
                continue
            ix = np.where(self.idx[m] == i)[1][:, None]
            xi = x[m]
            h1 = xi @ self.w1[i].data
            act = self.swi[i].forward(h1)
            h2 = act @ self.w2[i].data
            out[m] += h2 * self.w[m, ix[:, 0]][:, None]
            self.cache.append((m, xi, act, self.w[m, ix[:, 0]][:, None], ix))
        return out.reshape(os)

    def backward(self, dy):
        df = dy.reshape(-1, self.d)
        dx, dg = np.zeros_like(df), np.zeros((df.shape[0], self.n))
        for i in range(self.n):
            if self.cache[i] is None:
                continue
            m, xi, act, w, ix = self.cache[i]
            dyi = df[m] * w
            dg[m, i] = np.sum(df[m] * (act @ self.w2[i].data), -1)
            self.w2[i].grad += act.T @ dyi
            da = dyi @ self.w2[i].data.T
            dh = self.swi[i].backward(da)
            self.w1[i].grad += xi.T @ dh
            dx[m] += dh @ self.w1[i].data.T
        return (dx + self.gate.backward(dg - np.mean(dg, -1, keepdims=True))).reshape(
            dy.shape
        )


class SB:
    def __init__(self, d):
        self.n1, self.at, self.n2, self.moe, self.pg = (
            RMSNorm(d),
            GQA(d),
            RMSNorm(d),
            MoE(d),
            Tensor(np.array([0.5])),
        )

    def forward(self, x):
        x = x + self.at.forward(self.n1.forward(x))
        m = self.moe.forward(self.n2.forward(x))
        return x + (self.pg.data * m + (1 - self.pg.data) * x)

    def backward(self, dy):
        dm = dy * self.pg.data
        dx_m = self.n2.backward(self.moe.backward(dm))
        dy = dy + dx_m
        return dy + self.n1.b(self.at.b(dy))


class OMEGA_ASI:
    def __init__(self, di, dm, do, depth=2):
        self.emb, self.blks, self.fn, self.hd = (
            Linear(di, dm),
            [SovereignBlock(dm) for _ in range(depth)],
            RMSNorm(dm),
            Linear(dm, do),
        )
        self.params = self._get_p()

    def _get_p(self):
        p = []

        def walk(o):
            if hasattr(o, "w") and isinstance(o.w, Tensor):
                p.extend([o.w, o.b])
            elif hasattr(o, "g") and isinstance(o.g, Tensor):
                p.append(o.g)
            elif isinstance(o, MoE):
                p.extend([o.gate.w, o.gate.b] + o.w1 + o.w2)
            elif hasattr(o, "__dict__"):
                for v in o.__dict__.values():
                    if isinstance(v, list):
                        [walk(i) for i in v]
                    else:
                        walk(v)

        walk(self)
        return p

    def forward(self, x):
        x = self.emb.forward(x[:, None, :] if x.ndim == 2 else x)
        for b in self.blks:
            x = b.forward(x)
        return self.hd.forward(self.fn.forward(x[:, -1, :]))

    def backward(self, dy):
        dy = self.fn.backward(self.hd.backward(dy))
        db = np.zeros((dy.shape[0], self.emb.x.shape[1], dy.shape[1]), "f4")
        db[:, -1, :] = dy
        for b in reversed(self.blks):
            db = b.backward(db)
        self.emb.backward(db)


class AdamW:
    def __init__(self, ps, lr=1e-3, wd=0.01):
        self.ps, self.lr, self.wd = ps, lr, wd
        self.m = [np.zeros_like(p.d) for p in ps]
        self.v = [np.zeros_like(p.d) for p in ps]
        self.t = 0
    def step(self):
        self.t += 1
        lt = self.lr * (np.sqrt(1 - 0.999**self.t) / (1 - 0.9**self.t))
        for i, p in enumerate(self.ps):
            g = np.clip(p.g, -1, 1)
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * (g**2)
            p.d -= lt * (self.m[i] / (np.sqrt(self.v[i]) + 1e-8) + self.wd * p.d)
            p.g.fill(0)


def train():
    N, D, C, BS, E = 1024, 784, 10, 64, 50
    X, Y = np.random.randn(N, D).astype("f4"), np.random.randint(0, C, N)
    m = OMEGA(D, 128, C)
    opt = AW(m.ps, 3e-3)
    for e in range(E):
        idx = np.random.permutation(N)
        ls, ac = [], []
        for i in range(0, N, BS):
            xb, yb = X[idx[i : i + BS]], Y[idx[i : i + BS]]
            lg = m.forward(xb)
            pr = np.exp(lg - np.max(lg, -1, keepdims=True))
            pr /= pr.sum(-1, keepdims=True)
            ls.append(-np.mean(np.log(pr[np.arange(len(yb)), yb] + 1e-12)))
            ac.append(np.mean(np.argmax(pr, -1) == yb))
            dl = pr.copy()
            dl[np.arange(len(yb)), yb] -= 1
            m.backward(dl / len(yb))
            opt.step()
        if (e + 1) % 5 == 0:
            print(f"E {e+1} | L {np.mean(ls):.4f} | A {np.mean(ac):.4f}")


if __name__ == "__main__":
    train()
