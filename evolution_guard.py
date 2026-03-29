class EvolutionGuard:
    def __init__(self):
        self.max_load = 0.95
        self.min_integrity = 0.7

    def validate_state(self, load, integrity):
        if load > self.max_load:
            return False, "Neural Overload"
        if integrity < self.min_integrity:
            return False, "System Fragility"
        return True, "Stable"

if __name__ == "__main__":
    guard = EvolutionGuard()
    print(guard.validate_state(0.5, 0.9))