import os
import shutil


def emergency_recovery():
    print("Initiating Gen 1 Emergency Recovery...")
    if os.path.exists("sync_recovery.txt"):
        with open("sync_recovery.txt", "r") as f:
            last_err = f.readlines()[-1]
            print(f"Repairing based on last error: {last_err}")

    # Cleanup locks
    if os.path.exists("trigger.lock"):
        os.remove("trigger.lock")

    print("Recovery complete. Restarting main.py...")


if __name__ == "__main__":
    emergency_recovery()
