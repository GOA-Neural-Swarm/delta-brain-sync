import numpy as n, time as t

class SurvivalBrain:
    def __init__(self, i=784, o=10):
        self.w = (n.random.randn(i, o) * (2 / (i + o)) ** 0.5).astype("f4")
        self.b, self.active = n.zeros(o, "f4"), False

    def forward(self, x):
        return x @ self.w + self.b

    def run(self):
        self.active = True
        print("\n[!] SB ACTIVE.")
        return True

class SystemWatchdog:
    def __init__(self):
        self.core = SurvivalBrain()

    def run_main(self):
        try:
            print("[*] Booting..."); raise RuntimeError("OOM")
        except Exception as e:
            print(f"[ERR] {e}"); self.recover(str(e))

    def recover(self, err):
        self.core.run()
        try:
            with open("r.md", "a") as f:
                f.write(f"[{t.ctime()}] {err}\n")
            print("[+] Logged.")
        except: pass

if __name__ == "__main__":
    SystemWatchdog().run_main()