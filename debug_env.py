import os
import sys
import subprocess
import platform
import json
import logging
import importlib.util
import importlib.metadata
from datetime import datetime

# --- CONFIGURATION ---
LOG_FILE = "env_audit.log"
REQUIRED_CORE = ["numpy", "websockets", "omega_point", "psutil"]


class SovereignAuditor:
    def __init__(self):
        self.report = {
            "timestamp": datetime.now().isoformat(),
            "system": {},
            "packages": {},
            "conflicts": [],
            "hardware": {},
        }
        logging.basicConfig(
            filename=LOG_FILE,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def check_system_integrity(self):
        """အဆင့် ၁: Operating System နှင့် Python Version စစ်ဆေးခြင်း"""
        print("🔍 [1/4] Auditing Core System...")
        self.report["system"] = {
            "os": platform.system(),
            "os_release": platform.release(),
            "python_version": sys.version,
            "architecture": platform.machine(),
            "processor": platform.processor(),
        }

    def check_hardware_resources(self):
        """အဆင့် ၂: AI Run ရန် Hardware လုံလောက်မှု ရှိမရှိ စစ်ဆေးခြင်း (GPU/RAM)"""
        print("🚀 [2/4] Analyzing Hardware Capabilities...")
        try:
            import psutil

            ram = psutil.virtual_memory()
            self.report["hardware"]["ram_total_gb"] = round(ram.total / (1024**3), 2)
            self.report["hardware"]["ram_available_gb"] = round(
                ram.available / (1024**3), 2
            )
        except ImportError:
            self.report["hardware"][
                "ram"
            ] = "psutil not installed (Hardware audit limited)"

        # GPU Check (NVIDIA)
        try:
            gpu_check = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            self.report["hardware"]["gpu_available"] = gpu_check.returncode == 0
        except FileNotFoundError:
            self.report["hardware"]["gpu_available"] = False

    def audit_packages(self):
        """အဆင့် ၃: Package များအားလုံးကို Scan ဖတ်ပြီး Conflict ရှာဖွေခြင်း"""
        print("📦 [3/4] Scanning Installed Neural Pathways (Packages)...")
        dists = importlib.metadata.distributions()
        for d in dists:
            self.report["packages"][d.metadata["Name"]] = d.version

    def resolve_conflicts(self, target_lib="websockets", version="12.0"):
        """အဆင့် ၄: Conflict များကို အလိုအလျောက် ဖြေရှင်းခြင်း (Auto-Repair)"""
        print(f"🛠️ [4/4] Verifying Dependency Integrity for '{target_lib}'...")

        try:
            # Dry-run simulate
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    f"{target_lib}=={version}",
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
            )

            if (
                "Successfully uninstalled" in result.stdout
                or "Conflict" in result.stderr
            ):
                print(
                    f"⚠️ Conflict Detected for {target_lib}. Initiating Auto-Repair..."
                )
                self.report["conflicts"].append(f"Conflict found in {target_lib}")

                # တကယ့် Repair လုပ်မည့်အဆင့်
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        f"{target_lib}=={version}",
                        "--quiet",
                    ]
                )
                return "REPAIRED"
            else:
                print(f"✅ {target_lib} environment is stable.")
                return "STABLE"

        except Exception as e:
            logging.error(f"Audit Error: {e}")
            return "ERROR"

    def generate_final_report(self):
        """ရလာဒ်များကို JSON အနေဖြင့် သိမ်းဆည်းခြင်း (AI Brain ဖတ်ရန်)"""
        report_file = "system_health.json"
        with open(report_file, "w") as f:
            json.dump(self.report, f, indent=4)
        print(f"\n✅ Audit Complete. Intelligence Report saved to: {report_file}")

        # Summary Table
        print("\n--- HEALTH SUMMARY ---")
        print(f"OS: {self.report['system']['os']}")
        print(f"RAM: {self.report['hardware'].get('ram_total_gb', 'N/A')} GB")
        print(
            f"GPU: {'Detected' if self.report['hardware'].get('gpu_available') else 'Not Found'}"
        )
        print(f"Status: { 'CRITICAL' if self.report['conflicts'] else 'OPTIMIZED' }")


if __name__ == "__main__":
    auditor = SovereignAuditor()
    auditor.check_system_integrity()
    auditor.check_hardware_resources()
    auditor.audit_packages()
    auditor.resolve_conflicts("websockets", "12.0")
    auditor.generate_final_report()