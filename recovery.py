import os
def recover_from_failure():
    print("🛠️ [RECOVERY]: Cleaning system locks...")
    if os.path.exists("agi_system.db-journal"):
        os.remove("agi_system.db-journal")