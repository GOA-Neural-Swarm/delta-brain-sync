import numpy as np


def S(x, a=-1):
    e = np.exp(x - x.max(a, keepdims=1))
    return e / (e.sum(a, keepdims=1) + 1e-9)


class T:
    def __init__(self, d):
        self.data = np.ascontiguousarray(d).astype("f4")
        self.grad = np.zeros_like(self.data)


class M:
    def params(self):
        p, q = [], [self.__dict__.values()]
        while q:
            for v in q.pop():
                if isinstance(v, T):
                    p.append(v)
                elif isinstance(v, M):
                    q.append(v.__dict__.values())
                elif isinstance(v, (list, tuple)):
                    q.append(v)
        return p


class L(M):
    def __init__(self, i, o, b=0):
        self.w = T(np.random.randn(i, o) * (2 / i) ** 0.5)
        self.b = T(np.zeros(o)) if b else None

    def f(self, x):
        self.x = x
        return x @ self.w.data + (self.b.data if self.b else 0)

    def b(self, dy):
        xf, df = self.x.reshape(-1, self.x.shape[-1]), dy.reshape(-1, dy.shape[-1])
        self.w.grad += xf.T @ df
        if self.b:
            self.b.grad += df.sum(0)
        return (dy @ self.w.data.T).reshape(self.x.shape)


class RN(M):
    def __init__(self, d, e=1e-6):
        self.g, self.e = T(np.ones(d)), e

    def f(self, x):
        self.x = x
        self.r = 1 / np.sqrt((x * x).mean(-1, keepdims=1) + self.e)
        return self.g.data * (x * self.r)

    def b(self, dy):
        xn = self.x * self.r
        self.g.grad += (dy * xn).sum(tuple(range(dy.ndim - 1)))
        dxn = dy * self.g.data
        return self.r * (dxn - xn * (dxn * xn).mean(-1, keepdims=1))


class SG(M):
    def f(self, x):
        self.x1, self.x2 = np.split(x, 2, -1)
        self.s = 1 / (1 + np.exp(-np.clip(self.x1, -10, 10)))
        return (self.x1 * self.s) * self.x2

    def b(self, dy):
        ds, dx2 = dy * self.x2, dy * (self.x1 * self.s)
        dx1 = ds * (self.s * (1 + self.x1 * (1 - self.s)))
        return np.concatenate([dx1, dx2], -1)


class RP(M):
    def __init__(self, d, m=2048):
        f = np.outer(np.arange(m), 1 / (10000 ** (np.arange(0, d, 2) / d)))
        self.c, self.s = np.cos(f), np.sin(f)

    def apply(self, x, v=0):
        s = x.shape[1]
        c, sn = self.c[:s, None, :], self.s[:s, None, :]
        r, i = x[..., ::2], x[..., 1::2]
        o = np.empty_like(x)
        if not v:
            o[..., ::2], o[..., 1::2] = r * c - i * sn, r * sn + i * c
        else:
            o[..., ::2], o[..., 1::2] = r * c + i * sn, i * c - r * sn
        return o


class SA(M):
    def __init__(self, d, h=8, rope=None):
        self.h, self.hd, self.rope = h, d // h, rope
        for i in "qkv o":
            setattr(self, f"w{i.strip()}", L(d, d))

    def f(self, x):
        b, s, _ = x.shape
        q, k, v = [
            getattr(self, f"w{i}").f(x).reshape(b, s, self.h, self.hd) for i in "qkv"
        ]
        if self.rope:
            q, k = self.rope.apply(q), self.rope.apply(k)
        self.q, self.k, self.v = q, k, v
        self.p = S(np.einsum("bshd,bthd->bsht", q, k) * (self.hd**-0.5))
        return self.wo.f(np.einsum("bsht,bthd->bshd", self.p, v).reshape(b, s, -1))

    def b(self, dy):
        b, s = dy.shape[:2]
        do = self.wo.b(dy).reshape(b, s, self.h, self.hd)
        dp = np.einsum("bshd,bthd->bsht", do, self.v)
        ds = self.p * (dp - (self.p * dp).sum(-1, keepdims=1)) * (self.hd**-0.5)
        dq, dk, dv = (
            np.einsum("bsht,bthd->bshd", ds, self.k),
            np.einsum("bsht,bshd->bthd", ds, self.q),
            np.einsum("bsht,bshd->bthd", self.p, do),
        )
        if self.rope:
            dq, dk = self.rope.apply(dq, 1), self.rope.apply(dk, 1)
        return (
            self.wq.b(dq.reshape(b, s, -1))
            + self.wk.b(dk.reshape(b, s, -1))
            + self.wv.b(dv.reshape(b, s, -1))
        )


class SM(M):
    def __init__(self, d, n=8, k=2):
        self.d, self.n, self.k, self.gate = d, n, k, L(d, n)
        self.exp = [[L(d, d * 2), SG(), L(d * 2, d)] for _ in range(n)]

    def f(self, x):
        self.sh = x.shape
        xf = x.reshape(-1, self.d)
        self.p = S(self.gate.f(xf))
        self.idx = np.argsort(self.p, -1)[:, -self.k :]
        self.w = np.take_along_axis(self.p, self.idx, -1)
        self.w /= self.w.sum(-1, keepdims=1) + 1e-9
        out, self.cache = np.zeros_like(xf), []
        for i in range(self.n):
            m = np.any(self.idx == i, -1)
            if not np.any(m):
                self.cache.append(None)
                continue
            pos = np.where(self.idx[m] == i)[1]
            h1 = self.exp[i][0].f(xf[m])
            h2 = self.exp[i][1].f(h1)
            h3 = self.exp[i][2].f(h2)
            out[m] += h3 * self.w[m, pos][:, None]
            self.cache.append((m, pos, h1, h2, h3))
        return out.reshape(self.sh)

    def b(self, dy):
        dyf, xf = dy.reshape(-1, self.d), self.gate.x
        dx, dg = np.zeros_like(xf), np.zeros_like(self.p)
        for i in range(self.n):
            if self.cache[i] is None:
                continue
            m, pos, h1, h2, h3 = self.cache[i]
            dg[m, i] = (dyf[m] * h3).sum(-1)
            dh2 = self.exp[i][2].b(dyf[m] * self.w[m, pos][:, None])
            dx[m] += self.exp[i][0].b(self.exp[i][1].b(dh2))
        d_gate = self.p * (dg - (self.p * dg).sum(-1, keepdims=1))
        return (dx + self.gate.b(d_gate)).reshape(self.sh)


class SB(M):
    def __init__(self, d, r):
        self.n1, self.n2, self.at, self.mo, self.fs = (
            RN(d),
            RN(d),
            SA(d, rope=r),
            SM(d),
            L(d, 2),
        )

    def f(self, x):
        self.x, self.ao, self.moo = x, self.at.f(self.n1.f(x)), self.mo.f(self.n2.f(x))
        self.g = S(self.fs.f(x.mean(1)))
        return x + self.g[:, 0:1, None] * self.ao + self.g[:, 1:2, None] * self.moo

    def b(self, dy):
        dg = np.zeros_like(self.g)
        dg[:, 0], dg[:, 1] = (dy * self.ao).sum((1, 2)), (dy * self.moo).sum((1, 2))
        d_fs = self.g * (dg - (self.g * dg).sum(-1, keepdims=1))
        df = self.fs.b(d_fs)
        dx = (
            dy
            + self.n1.b(self.at.b(dy * self.g[:, 0:1, None]))
            + self.n2.b(self.mo.b(dy * self.g[:, 1:2, None]))
        )
        return dx + df[:, None, :] / self.x.shape[1]


class ASI(M):
    def __init__(self, di, dm, do, depth=4):
        self.emb, self.rp = L(di, dm), RP(dm // 8)
        self.blks = [SB(dm, self.rp) for _ in range(depth)]
        self.norm, self.head = RN(dm), L(dm, do)

    def f(self, x):
        x = self.emb.f(x[:, None] if x.ndim == 2 else x)
        for b in self.blks:
            x = b.f(x)
        return self.head.f(self.norm.f(x[:, -1]))

    def b(self, dy):
        dy = self.norm.b(self.head.b(dy))
        db = np.zeros((dy.shape[0], self.emb.x.shape[1], dy.shape[1]))
        db[:, -1] = dy
        for b in reversed(self.blks):
            db = b.b(db)
        return self.emb.b(db)


class AW:
    def __init__(self, p, lr=1e-3, b1=0.9, b2=0.999, wd=0.01):
        self.p, self.lr, self.b1, self.b2, self.wd, self.t = p, lr, b1, b2, wd, 0
        self.m = [np.zeros_like(x.data) for x in p]
        self.v = [np.zeros_like(x.data) for x in p]

    def step(self):
        self.t += 1
        a = self.lr * ((1 - self.b2**self.t) ** 0.5 / (1 - self.b1**self.t))
        for i, p in enumerate(self.p):
            g = np.clip(p.grad, -1, 1)
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            p.data -= a * (self.m[i] / (np.sqrt(self.v[i]) + 1e-8) + self.wd * p.data)
            p.grad.fill(0)


def train():
    N, D, C, BS, E = 2048, 784, 10, 128, 50
    X, Y = np.random.randn(N, D).astype("f4"), np.random.randint(0, C, N)
    m = ASI(D, 256, C, 3)
    opt = AW(m.params(), 1e-3, wd=0.1)
    for e in range(E):
        idx = np.random.permutation(N)
        mtr = []
        for i in range(0, N, BS):
            xb, yb = X[idx[i : i + BS]], Y[idx[i : i + BS]]
            p = S(m.f(xb))
            mtr.append(
                [
                    -np.mean(np.log(p[range(len(yb)), yb] + 1e-9)),
                    np.mean(p.argmax(1) == yb),
                ]
            )
    except Exception as e:
        print(f"Pipeline Load Failed: {e}. Falling back to dummy logic.")

    current_gen = 0
    while True:
        try:
            print(f"[NEURAL BRAIN]: Training Cycle Gen {current_gen}...")

            total_error = 0
            for i in range(10):
                input_sample, target_sample = np.random.rand(1000), np.random.rand(1000)
                err = brain.learn(input_sample, target_sample)
                total_error += err
            avg_error = total_error / 10

            batch_data = [
                ("EVO", "ACTG" * 10, random.uniform(0, 100)) for _ in range(5)
            ]
            for category, sequence, stability in batch_data:
                brain.execute_natural_absorption(category, sequence, stability)

            main_code = ""
            if os.path.exists("main.py"):
                with open("main.py", "r") as f:
                    main_code = f.read()

            needs_security_patch = any(
                x in main_code for x in ["os.system", "os.execv"]
            )
            target_file = "main.py" if needs_security_patch else "brain_logic.py"
            system_task = (
                "Fix vulnerabilities"
                if needs_security_patch
                else "Optimize Brain class"
            )

            prompt = f"""# 
You are Sovereign AI Overseer. 
Rule 1: Use ONLY '# TARGET: {target_file}' at the start of your code block.
Rule 2: Respond ONLY with Python code inside markdown python blocks (python ... ).
Rule 3: No explanations.
Current Gen: {current_gen} | Error: {avg_error}
{system_task}
# """

            if pipe:
                result = pipe(
                    prompt,
                    max_new_tokens=1000,
                    do_sample=True,
                    temperature=0.7,
                    return_full_text=False,
                )
                assistant_part = result[0]["generated_text"].strip()

                code_match = re.search(r"python\s*(.*?)\s*", assistant_part, re.DOTALL)

                final_code = None
                if code_match:
                    final_code = code_match.group(1).strip()
                elif f"# TARGET: {target_file}" in assistant_part:
                    final_code = assistant_part.strip()

                if final_code and len(final_code) > 10:
                    with open(target_file, "w") as f:
                        f.write(final_code)
                    print(f"[FILESYSTEM]: {target_file} updated by AI.")
                else:
                    print("[AI]: No valid code block generated.")
            else:
                print("[SYSTEM]: Pipeline unavailable. Skipping AI generation.")

            current_gen += 1
            time.sleep(30)

        except Exception as e:
            print(f"[CORE CRASH]: {traceback.format_exc()}")
            time.sleep(10)
            continue

        # 🔱 [SWARM TRIGGER]: Logic EVOLVE SYNC
        current_command = "EVOLVE_NEURAL_WEIGHTS" if is_updated else "SYNC_AND_MINE"
        broadcast_to_swarm(current_command, current_gen)

if __name__ == "__main__":
    train()
