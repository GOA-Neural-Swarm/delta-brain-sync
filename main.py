import numpy as np

class Opt:
    def __init__(self, p):
        self.p, self.t = p, 0
        self.m, self.v = [x*0 for x in p], [x*0 for x in p]
    def step(self, g):
        self.t += 1
        r = 2e-3 * (1-.999**self.t)**.5 / (1-.9**self.t)
        for i, p in enumerate(self.p):
            self.m[i] = .9*self.m[i] + .1*g[i]
            self.v[i] = .999*self.v[i] + .001*g[i]**2
            p *= .99998; p -= r * self.m[i] / (self.v[i]**.5 + 1e-8)

class LN:
    def __init__(self, d): self.w, self.b = np.ones((1,d),'f4'), np.zeros((1,d),'f4')
    def forward(self, x):
        self.h = (x-(m:=x.mean(-1,keepdims=1))) / (self.s:=np.sqrt(x.var(-1,keepdims=1)+1e-5))
        return self.w*self.h + self.b
    def backward(self, d):
        dx = d*self.w; self.dw, self.db = (d*self.h).sum(0,keepdims=1), d.sum(0,keepdims=1)
        return (dx - dx.mean(-1,keepdims=1) - self.h*(dx*self.h).mean(-1,keepdims=1)) / self.s

class S:
    def forward(self, x): self.x, self.s = x, 1/(1+np.exp(-np.clip(x,-20,20))); return x*self.s
    def backward(self, d): return d*(self.s + self.x*self.s*(1-self.s))

class L:
    def __init__(self, i, o): self.w, self.b = np.random.randn(i,o).astype('f4')*(2/i)**.5, np.zeros((1,o),'f4')
    def forward(self, x): self.x=x; return x@self.w + self.b
    def backward(self, d): self.dw, self.db = self.x.T@d, d.sum(0,keepdims=1); return d@self.w.T

class B:
    def __init__(self, d): self.ly = [LN(d), L(d,d), S(), L(d,d)]
    def forward(self, x):
        h = x
        for l in self.ly: h = l.forward(h)
        return h + x
    def backward(self, d):
        for l in self.ly[::-1]: d = l.backward(d)
        return d + d # Residual grad

class Model:
    def __init__(self, i=784, h=128, o=10, n=3):
        self.bl = [L(i,h)] + [B(h) for _ in range(n)] + [L(h,o)]
        self.ly = [y for x in self.bl for y in (getattr(x,'ly',[x])) if hasattr(y,'w')]
        self.opt = Opt([p for y in self.ly for p in (y.w,y.b)])
    def forward(self, x):
        for l in self.bl: x = l.forward(x)
        return x
    def backward(self, d):
        for l in self.bl[::-1]: d = l.backward(d)
        self.opt.step([g for y in self.ly for g in (y.dw,y.db)])

def step(m, x, y):
    z = m.forward(x)
    p = (e:=np.exp(z-z.max(1,keepdims=1))) / e.sum(1,keepdims=1)
    l = -np.log(p[np.arange(len(y)),y]+1e-10).mean()
    dl = p.copy(); dl[np.arange(len(y)),y]-=1; m.backward(dl/len(y))
    return l

X, Y = np.random.randn(100,784).astype('f4'), np.random.randint(0,10,100)
m = Model()
for e in range(101):
    l = step(m, X, Y)
    if e%10==0: print(f"E:{e:03}|L:{l:.4f}|A:{(m.forward(X).argmax(1)==Y).mean():.4f}")