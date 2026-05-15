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
CORE_VERSIONS = {
    "numpy": "1.23.4",
    "websockets": "12.0",
    "omega_point": "0.1.0",
    "psutil": "5.9.3",
}


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
        print(" [1/5] Auditing Core System...")
        self.report["system"] = {
            "os": platform.system(),
            "os_release": platform.release(),
            "python_version": sys.version,
            "architecture": platform.machine(),
            "processor": platform.processor(),
        }

    def check_hardware_resources(self):
        print(" [2/5] Analyzing Hardware Capabilities...")
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

        try:
            gpu_check = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            self.report["hardware"]["gpu_available"] = gpu_check.returncode == 0
        except FileNotFoundError:
            self.report["hardware"]["gpu_available"] = False

    def audit_packages(self):
        print(" [3/5] Scanning Installed Neural Pathways (Packages)...")
        dists = importlib.metadata.distributions()
        for d in dists:
            self.report["packages"][d.metadata["Name"]] = d.version

    def resolve_conflicts(self):
        print(" [4/5] Resolving Package Conflicts...")
        for package, version in CORE_VERSIONS.items():
            try:
                importlib.import_module(package)
            except ImportError:
                self.report["conflicts"].append(f"{package} not installed")
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        f"{package}=={version}",
                        "--quiet",
                    ]
                )
                self.report["conflicts"].append(f"{package} installed")
                continue

            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        f"{package}=={version}",
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
                        f" Conflict Detected for {package}. Initiating Auto-Repair..."
                    )
                    self.report["conflicts"].append(f"Conflict found in {package}")

                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "pip",
                            "install",
                            f"{package}=={version}",
                            "--quiet",
                        ]
                    )
                    self.report["conflicts"].append(f"{package} repaired")
                else:
                    print(f" {package} environment is stable.")
            except Exception as e:
                logging.error(f"Audit Error: {e}")
                self.report["conflicts"].append(f"Error occurred for {package}")

    def check_additive_evolution(self):
        print(" [5/5] Checking Additive Evolution...")
        for package in CORE_VERSIONS:
            try:
                importlib.import_module(package)
                package_version = importlib.metadata.version(package)
                if package_version != CORE_VERSIONS[package]:
                    print(
                        f" Package {package} has outdated version: {package_version}. Updating to {CORE_VERSIONS[package]}..."
                    )
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "pip",
                            "install",
                            f"{package}=={CORE_VERSIONS[package]}",
                            "--quiet",
                        ]
                    )
            except ImportError:
                print(f" Package {package} not installed. Installing...")
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        f"{package}=={CORE_VERSIONS[package]}",
                        "--quiet",
                    ]
                )

    def generate_final_report(self):
        report_file = "system_health.json"
        with open(report_file, "w") as f:
            json.dump(self.report, f, indent=4)
        print(f"\n Audit Complete. Intelligence Report saved to: {report_file}")

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
    auditor.resolve_conflicts()
    auditor.check_additive_evolution()
    auditor.generate_final_report()