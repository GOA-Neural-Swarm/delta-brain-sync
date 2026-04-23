import numpy as n, time as t
class S:
 def __init__(self, i=784, o=10): self.w, self.b, self.a = n.random.randn(i, o).astype("f4") * (2 / (i + o)) ** .5, n.zeros(o, "f4"), 0
 def r(self): self.a = 1; print("\n[!] SB ACTIVE.")
class W:
 def __init__(self): self.c = S()
 def m(self):
  print("[*] Booting...")
  try: raise RuntimeError("OOM")
  except Exception as e:
   s = str(e); print(f"[ERR] {s}"); self.c.r()
   try:
    with open("r.md", "a") as f: f.write(f"[{t.ctime()}] {s}\n")
    print("[+] Logged.")
   except: pass
if __name__ == "__main__": W().m()