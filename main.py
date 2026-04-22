import numpy as n, time as t

class S:
    def __init__(self, i=784, o=10):
        self.w = (n.random.randn(i, o) * (2 / (i + o)) ** 0.5).astype("f4")
        self.b, self.a = n.zeros(o, "f4"), 0
    def f(self, x): return x @ self.w + self.b
    def r(self): 
        self.a = 1
        print("\n[!] SB ACTIVE.")
        return 1

class W:
    def __init__(self): self.c = S()
    def m(self):
        try:
            print("[*] Booting...")
            raise RuntimeError("OOM")
        except Exception as e:
            print(f"[ERR] {e}"); self.v(str(e))
    def v(self, e):
        self.c.r()
        try:
            with open("r.md", "a") as f: f.write(f"[{t.ctime()}] {e}\n")
            print("[+] Logged.")
        except: pass

if __name__ == "__main__":
    W().m()