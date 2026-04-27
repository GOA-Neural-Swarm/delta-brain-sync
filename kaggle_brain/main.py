import subprocess
import sys
import importlib
import pkg_resources


def install_requirements():
    libs = [
        "psycopg2-binary",
        "firebase-admin",
        "bitsandbytes",
        "requests",
        "accelerate",
        "GitPython",
        "sympy==1.12",
        "numpy",
        "scikit-learn",
        "transformers",
        "huggingface-hub",
    ]
    try:
        for lib in libs:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    lib,
                    "--quiet",
                    "--no-cache-dir",
                    "--upgrade",
                ]
            )
        print("\u2705 [SYSTEM]: Phase 7.1 Sovereign Core & Stability Patch Ready.")
    except subprocess.CalledProcessError as e:
        print(f"\u26a0\ufe0f Install Warning: Error installing requirements: {e}")
    except Exception as e:
        print(f"\u26a0\ufe0f Install Warning: An unexpected error occurred: {e}")


def check_requirements():
    required_libs = [
        "psycopg2-binary",
        "firebase-admin",
        "bitsandbytes",
        "requests",
        "accelerate",
        "GitPython",
        "sympy",
        "numpy",
        "scikit-learn",
        "transformers",
        "huggingface-hub",
    ]
    installed_libs = pkg_resources.working_set
    installed_libs_names = [lib.project_name for lib in installed_libs]
    missing_libs = [lib for lib in required_libs if lib not in installed_libs_names]
    return missing_libs


def main():
    missing_libs = check_requirements()
    if missing_libs:
        print(
            f"\u26a0\ufe0f Install Warning: Missing required libraries: {missing_libs}"
        )
        install_requirements()
    else:
        print("\u2705 [SYSTEM]: All required libraries are installed.")


if __name__ == "__main__":
    main()
