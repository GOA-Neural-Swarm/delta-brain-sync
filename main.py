import numpy as n
k={'keepdims':1}
class O:
 def __init__(self,p):self.p,self.t,self.m,self.v=p,0,[0*x for x in p],[0*x for x in p]
 def s(self,g):
  self.t+=1;r=2e-3*(1-.999**self.t)**.5/(1-.9**self.t)
  for p,m,v,g in zip(self.p,self.m,self.v,g):m[:]=.9*m+.1*g;v[:]=.999*v+.001*g**2;p*=.99998;p-=r*m/(v**.5+1e-8)
class N:
 def __init__(self,d):self.w,self.b=n.ones((1,d),'f4'),n.zeros((1,d),'f4')
 def f(self,x):self.h=(x-x.mean(-1,**k))/(self.v:=(x.var(-1,**k)+1e-5)**.5);return self.w*self.h+self.b
 def r(self,d):x=d*self.w;self.dw,self.db=(d*self.h).sum(0,**k),d.sum(0,**k);return(x-x.mean(-1,**k)-self.h*(x*self.h).mean(-1,**k))/self.v
class A:
 def f(self,x):self.x=x;self.s=1/(1+n.exp(-n.clip(x,-20,20)));return x*self.s
 def r(self,d):return d*(self.s+self.x*self.s*(1-self.s))
class L:
 def __init__(self,i,o):self.w,self.b=n.random.randn(i,o).astype('f4')*(2/i)**.5,n.zeros((1,o),'f4')
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
 def __init__(self,i=784,h=128,o=10,c=3):
  self.l=[L(i,h),*[B(h) for _ in range(c)],L(h,o)];self.y=[y for x in self.l for y in getattr(x,'l',[x]) if hasattr(y,'w')]
  self.o=O([p for x in self.y for p in (x.w,x.b)])
 def f(self,x):
  for l in self.l:x=l.f(x)
  return x
 def b(self,d):
  for l in self.l[::-1]:d=l.r(d)
  self.o.s([g for x in self.y for g in (x.dw,x.db)])
X,Y=n.random.randn(100,784).astype('f4'),n.random.randint(10,size=100)
m,U=M(),len(Y);I=n.arange(U)
for e in range(101):
 z=m.f(X);v=n.exp(z-z.max(1,**k));p=v/v.sum(1,**k);ls=-n.log(p[I,Y]+1e-10).mean();d=p.copy();d[I,Y]-=1;m.b(d/U)
 if not e%10:print(f"E:{e:03}|L:{ls:.4f}|A:{(z.argmax(1)==Y).mean():.4f}")