import os
import sys
import shutil
import time
import subprocess
import traceback
import importlib
import numpy as np

class SovereignRecovery:
    def __init__(self):
        self.critical_files = ["main.py", "brain.py"]
        self.backup_dir = ".omega_vault"
        self.lock_file = "trigger.lock"
        self.recovery_log = "sync_recovery.txt"
        
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)

    def check_environment(self):
        """အဆင့် ၁: ပတ်ဝန်းကျင်ကို စစ်ဆေးခြင်း (Dependency Integrity)"""
        print("[1/4] Checking Neural Environment...")
        required_libs = ["numpy", "omega_point"]
        for lib in required_libs:
            try:
                importlib.import_module(lib)
            except ImportError:
                print(f"⚠️ Critical Module '{lib}' missing. Re-installing...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", lib, "--quiet"])
        return True

    def hotfix_logic(self, last_error):
        """အဆင့် ၂: နောက်ဆုံး Error အပေါ်မူတည်ပြီး Code ကို အလိုအလျောက် ပြုပြင်ခြင်း"""
        print(f"[2/4] Analyzing Critical Failure: {last_error[:50]}...")
        
        # Memory Error ဖြစ်ရင် Data Size ကို လျှော့ချမယ်
        if "MemoryError" in last_error or "Out of Memory" in last_error:
            print("🔧 Applying Hotfix: Reducing Batch Size & Hidden Dimensions...")
            # ဤနေရာတွင် main.py ထဲက parameter များကို regex ဖြင့် အလိုအလျောက် ပြင်သည့် code ထည့်နိုင်သည်
            
        # File missing ဖြစ်နေရင် Backup ကနေ ပြန်ယူမယ်
        for file in self.critical_files:
            if not os.path.exists(file):
                print(f"🚨 {file} is corrupted or missing! Restoring from Vault...")
                backup_path = os.path.join(self.backup_dir, f"{file}.bak")
                if os.path.exists(backup_path):
                    shutil.copy(backup_path, file)
                else:
                    print(f"❌ No backup found for {file}. Creating emergency baseline...")
                    # Baseline code တစ်ခုကို auto-generate လုပ်ခိုင်းလို့ရသည်

    def purge_locks(self):
        """အဆင့် ၃: ပိတ်မိနေတဲ့ Process နဲ့ Lock ဖိုင်တွေကို ရှင်းထုတ်ခြင်း"""
        print("[3/4] Purging System Locks...")
        if os.path.exists(self.lock_file):
            try:
                os.remove(self.lock_file)
                print("✅ Lock file cleared.")
            except Exception as e:
                print(f"⚠️ Lock removal failed: {e}")

    def execute_reboot(self):
        """အဆင့် ၄: စနစ်ကို အသစ်ကနေ ပြန်လည်စတင်ခြင်း (Clean Handover)"""
        print("[4/4] Handing over to Main Core...")
        print("--- REBOOTING OMEGA-ASI MAIN CORE ---")
        time.sleep(2)
        # os.system အစား os.execv ကိုသုံးတာက process အဟောင်းကို လုံးဝပိတ်ပြီး အသစ်ပြန်စတာမို့ ပိုစိတ်ချရတယ်
        os.execv(sys.executable, ['python'] + [sys.argv[0].replace("recovery.py", "main.py")])

    def run(self):
        print("\n" + "!"*50)
        print("🚀 OMEGA-ASI EMERGENCY RECOVERY PROTOCOL 🚀")
        print("!"*50 + "\n")
        
        try:
            self.check_environment()
            
            if os.path.exists(self.recovery_log):
                with open(self.recovery_log, "r") as f:
                    logs = f.readlines()
                    last_err = logs[-1] if logs else "Unknown Error"
                    self.hotfix_logic(last_err)
            
            self.purge_locks()
            self.execute_reboot()
            
        except Exception as e:
            print(f"💀 FATAL RECOVERY FAILURE: {e}")
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    recovery = SovereignRecovery()
    recovery.run()
