import subprocess
import sys
import pkg_resources

print("🔬 [DIAGNOSIS]: Starting Environment Audit...")
print(f"Python Version: {sys.version}")

print("\n📦 [INSTALLED PACKAGES]:")
installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
for name, version in sorted(installed_packages.items()):
    print(f"{name} == {version}")

print("\n⚠️ [CONFLICT CHECK]: Checking websockets dependencies...")
try:
    import websockets

    print(f"✅ websockets is already installed. Version: {websockets.__version__}")
except ImportError:
    print("❌ websockets is NOT installed.")

# Check why pip is failing
print("\n🛠️ [PIP TEST]: Simulating dry-run installation...")
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "websockets==12.0", "--dry-run"],
    capture_output=True,
    text=True,
)
print(result.stdout)
print(result.stderr)
