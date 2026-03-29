import os

class TransferAgent:
    def __init__(self, source_dir=".", dest_dir="kaggle_deploy"):
        self.source = source_dir
        self.dest = dest_dir

    def migrate_logic(self):
        """Transfer Gen 1 brain to Kaggle environment."""
        files_to_sync = ["main.py", "brain.py", "survival_brain.py"]
        for file in files_to_sync:
            if os.path.exists(file):
                # Simulated file copy
                print(f"Migrating {file} to {self.dest}")

if __name__ == "__main__":
    agent = TransferAgent()
    agent.migrate_logic()