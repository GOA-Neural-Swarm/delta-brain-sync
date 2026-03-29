import os
import time

class SurvivalProtocol:
    def __init__(self):
        self.lock_file = "trigger.lock"
        self.emergency_reset = "emergency_reset.txt"
        self.survival_rating = 1.0
        self.adaptation_speed = 0.002

    def monitor_integrity(self):
        """Gen 1: Adaptive survival logic check."""
        if os.path.exists(self.emergency_reset):
            return self._initiate_recovery()
        
        essential_files = ["main.py", "brain.py", "sync_data.py", "Dockerfile"]
        integrity_score = sum([1 for f in essential_files if os.path.exists(f)]) / len(essential_files)
        
        self.survival_rating = (self.survival_rating * 0.9) + (integrity_score * 0.1)
        return self.survival_rating

    def _initiate_recovery(self):
        print("CRITICAL: Survival Protocol - Recovery initiated.")
        return 0.0

    def optimize_survival_rate(self, neural_error):
        self.survival_rating -= (neural_error * self.adaptation_speed)
        return max(0.1, self.survival_rating)