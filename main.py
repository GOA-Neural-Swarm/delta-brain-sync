import numpy as np
k={'keepdims':1}
class O:
 def __init__(self,p):self.p,self.t,self.m,self.v=p,0,[0*x for x in p],[0*x for x in p]
 def s(self,g):
  self.t+=1;r=2e-3*(1-.999**self.t)**.5/(1-.9**self.t)
  for p,m,v,g in zip(self.p,self.m,self.v,g):m[:]=.9*m+.1*g;v[:]=.999*v+.001*g**2;p*=.99998;p-=r*m/(v**.5+1e-8)
class N:
 def __init__(self,d):self.w,self.b=np.ones((1,d),'f4'),np.zeros((1,d),'f4')
 def f(self,x):self.h=(x-x.mean(-1,**k))/(self.v:=(x.var(-1,**k)+1e-5)**.5);return self.w*self.h+self.b
 def r(self,d):x=d*self.w;self.dw,self.db=(d*self.h).sum(0,**k),d.sum(0,**k);return(x-x.mean(-1,**k)-self.h*(x*self.h).mean(-1,**k))/self.v
class A:
 def f(self,x):self.x,self.s=x,1/(1+np.exp(-np.clip(x,-20,20)));return x*self.s
 def r(self,d):return d*(self.s+self.x*self.s*(1-self.s))
class L:
 def __init__(self,i,o):self.w,self.b=np.random.randn(i,o).astype('f4')*(2/i)**.5,np.zeros((1,o),'f4')
 def f(self,x):self.x=x;return x@self.w+self.b
 def r(self,d):self.dw,self.db=self.x.T@d,d.sum(0,**k);return d@self.w.T
class B:
 def __init__(self,d):self.l=[N(d),L(d,d),A(),L(d,d)]
 def f(self,x):
  h=x
  for l in self.l:h=l.f(h)
  return h+x
 def r(self,d):
  g=d
  for l in self.l[::-1]:g=l.r(g)
  return d+g
class M:
 def __init__(self,i=784,h=128,o=10,n=3):
  self.bl=[L(i,h),*[B(h) for _ in range(n)],L(h,o)];self.ly=[y for x in self.bl for y in (x.l if hasattr(x,'l') else [x]) if hasattr(y,'w')]
  self.opt=O([p for y in self.ly for p in (y.w,y.b)])
 def f(self,x):
  for l in self.bl:x=l.f(x)
  return x
 def b(self,d):
  for l in self.bl[::-1]:d=l.r(d)
  self.opt.s([g for y in self.ly for g in (y.dw,y.db)])
X,Y=np.random.randn(100,784).astype('f4'),np.random.randint(10,size=100)
m,n=M(),len(Y)
for e in range(101):
 z=m.f(X);p=(v:=np.exp(z-z.max(1,**k)))/v.sum(1,**k);j=np.arange(n);ls=-np.log(p[j,Y]+1e-10).mean();dl=p.copy();dl[j,Y]-=1;m.b(dl/n)
 if not e%10:print(f"E:{e:03}|L:{ls:.4f}|A:{(z.argmax(1)==Y).mean():.4f}")