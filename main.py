import numpy as n
class N:
 def __init__(self,d):self.g,self.b=n.ones(d),n.zeros(d)
 def __call__(self,x):self.n=(x-x.mean(0))/n.sqrt(x.var(0)+1e-5);return self.n*self.g+self.b
 def back(self,d):self.dg=(d*self.n).sum(0);self.db=d.sum(0);return d*self.g
class A:
 def __call__(self,x):self.o=1/(1+n.exp(-n.clip(x,-20,20)));return self.o
 def back(self,d):return d*self.o*(1-self.o)
class L:
 def __init__(self,i,o):self.w,self.b=n.random.randn(i,o)*(2/i)**.5,n.zeros(o)
 def __call__(self,x):self.x=x;return x@self.w+self.b
 def back(self,d):self.dw,self.db=self.x.T@d,d.sum(0);return d@self.w.T
class B:
 def __init__(self,d,f):self.s=[N(d),L(d,d*f),A(),L(d*f,d)]
 def __call__(self,x):
  h=x
  for l in self.s:h=l(h)
  return h+x
 def back(self,d):
  g=d
  for l in reversed(self.s):g=l.back(g)
  return d+g
class M:
 def __init__(self,i=784,h=128,o=10,c=3):
  self.l,self.t=[L(i,h)]+[B(h,4)for _ in range(c)]+[B(h,2),B(h,3),L(h,o)],0;self.p=[]
  for l in self.l:
   for s in(l.s if hasattr(l,'s')else[l]):
    for a in('w','g','b'):
     if hasattr(s,a):self.p.append((s,a))
  self.m=[n.zeros_like(getattr(s,a))for s,a in self.p];self.v=[x*0 for x in self.m]
 def f(self,x):
  for l in self.l:x=l(x)
  return x
 def b(self,d):
  for l in reversed(self.l):d=l.back(d)
  self.t+=1;r=2e-3*(1-.999**self.t)**.5/(1-.9**self.t)
  for i,(s,a)in enumerate(self.p):
   g=getattr(s,'d'+a if a!='g'else'dg');self.m[i]=.9*self.m[i]+.1*g;self.v[i]=.999*self.v[i]+.001*g**2;setattr(s,a,getattr(s,a)-r*self.m[i]/(n.sqrt(self.v[i])+1e-8))
X,Y,m=n.random.randn(100,784).astype('f4'),n.random.randint(10,size=100),M()
for e in range(101):
 z=m.f(X);v=n.exp(z-z.max(1,keepdims=1));p=v/v.sum(1,keepdims=1);l,d=-n.log(p[range(100),Y]+1e-9).mean(),p.copy();d[range(100),Y]-=1;m.b(d)
 if e%10==0:print(f"E:{e} L:{l:.2f} A:{(z.argmax(1)==Y).mean():.2f}")