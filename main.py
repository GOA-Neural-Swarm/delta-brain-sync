import numpy as np
f32=np.float32
k={"keepdims":1}
class O:
 def __init__(self,p,lr=2e-3):self.p,self.lr,self.t,self.m,self.v=p,lr,0,[0]*len(p),[0]*len(p)
 def s(self,g):
  self.t+=1;a=self.lr*(1-.999**self.t)**.5/(1-.9**self.t)
  for i,p in enumerate(self.p):self.m[i]=.9*self.m[i]+.1*g[i];self.v[i]=.999*self.v[i]+.001*g[i]**2;p*=.99998;p-=a*self.m[i]/(self.v[i]**.5+1e-8)
class LN:
 def __init__(self,d):self.g,self.b=np.ones((1,d),f32),np.zeros((1,d),f32)
 def f(self,x):self.h=(x-x.mean(-1,**k))/(s:=np.sqrt(x.var(-1,**k)+1e-5));self.s=s;return self.g*self.h+self.b
 def bk(self,d):
  self.dg,self.db=(d*self.h).sum(0,**k),d.sum(0,**k);dx=d*self.g
  return(dx-dx.mean(-1,**k)-self.h*(dx*self.h).mean(-1,**k))/self.s
class SW:
 def f(self,x):self.x,self.s=x,1/(1+np.exp(-np.clip(x,-20,20)));return x*self.s
 def bk(self,d):return d*(self.s+self.x*self.s*(1-self.s))
class LI:
 def __init__(self,i,o):self.w,self.b=np.random.randn(i,o).astype(f32)*(2/i)**.5,np.zeros((1,o),f32)
 def f(self,x):self.x=x;return x@self.w+self.b
 def bk(self,d):self.dw,self.db=self.x.T@d,d.sum(0,**k);return d@self.w.T
class BL:
 def __init__(self,d):self.l=[LN(d),LI(d,d),SW(),LI(d,d)]
 def f(self,x):
  h=x
  for l in self.l:h=l.f(h)
  return h+x
 def bk(self,d):
  h=d
  for l in self.l[::-1]:h=l.bk(h)
  return h+d
class EN:
 def __init__(self,i=784,h=128,o=10,n=3):
  self.n=[LI(i,h)]+[BL(h) for _ in range(n)]+[LI(h,o)];self.o=[]
  for l in self.n:self.o+=getattr(l,'l', [l])
  self.o=[l for l in self.o if hasattr(l,'b')];self.p=[]
  for l in self.o:self.p+=[getattr(l,'w',getattr(l,'g',0)),l.b]
  self.u=O(self.p)
 def f(self,x):
  for l in self.n:x=l.f(x)
  return x
 def bk(self,d):
  for l in self.n[::-1]:d=l.bk(d)
  g=[]
  for l in self.o:g+=[getattr(l,'dw',getattr(l,'dg',0)),l.db]
  self.u.s(g)
def train():
 X,Y,m,r=np.random.randn(100,784).astype(f32),np.random.randint(0,10,100),EN(),np.arange(100)
 for e in range(101):
  z=m.f(X);ex=np.exp(z-z.max(1,**k));p=ex/ex.sum(1,**k);l=-np.log(p[r,Y]+1e-10).mean();dl=p.copy();dl[r,Y]-=1;m.bk(dl/100)
  if e%10==0:print(f"E:{e:03}|L:{l:.4f}|A:{(p.argmax(1)==Y).mean():.4f}")
if __name__=="__main__":train()