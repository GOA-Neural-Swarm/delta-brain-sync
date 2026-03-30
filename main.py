import numpy as np
k={'keepdims':1}
class N:
 def __init__(self,d):self.w,self.b=np.ones((1,d)),np.zeros((1,d))
 def f(self,x):self.h=(x-(m:=x.mean(-1,**k)))/(self.v:=(x.var(-1,**k)+1e-5)**.5);return self.w*self.h+self.b
 def r(self,d):x=d*self.w;self.dw,self.db=(d*self.h).sum(0,**k),d.sum(0,**k);return(x-x.mean(-1,**k)-self.h*(x*self.h).mean(-1,**k))/self.v
class A:
 def f(self,x):self.s=1/(1+np.exp(-np.clip(x,-20,20)));self.x=x;return x*self.s
 def r(self,d):return d*(self.s+self.x*self.s*(1-self.s))
class L:
 def __init__(self,i,o):self.w,self.b=np.random.randn(i,o)*(2/i)**.5,np.zeros((1,o))
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
  self.l=[L(i,h),*[B(h) for _ in range(c)],L(h,o)];self.y=[];self.t=0
  for l in self.l:
   for s in getattr(l,'l',[l]):
    if hasattr(s,'w'):self.y+=[s]
  self.m=[0*x.w for x in self.y]+[0*x.b for x in self.y];self.v=[0*x for x in self.m]
 def f(self,x):
  for l in self.l:x=l.f(x)
  return x
 def b(self,d,u):
  for l in self.l[::-1]:d=l.r(d)
  self.t+=1;r=2e-3*(1-.999**self.t)**.5/(1-.9**self.t);g=[x.dw/u for x in self.y]+[x.db/u for x in self.y];p=[x.w for x in self.y]+[x.b for x in self.y]
  for i in range(len(p)):self.m[i]=.9*self.m[i]+.1*g[i];self.v[i]=.999*self.v[i]+.001*g[i]**2;p[i][:]=p[i]*.99998-r*self.m[i]/(self.v[i]**.5+1e-8)
X,Y=np.random.randn(100,784).astype('f4'),np.random.randint(10,size=100)
m,U,I=M(),100,np.arange(100)
for e in range(101):
 z=m.f(X);v=np.exp(z-z.max(1,**k));p=v/v.sum(1,**k);ls=-np.log(p[I,Y]+1e-9).mean();d=p.copy();d[I,Y]-=1;m.b(d,U)
 if e%10==0:print(f"E:{e} L:{ls:.2f} A:{(z.argmax(1)==Y).mean():.2f}")