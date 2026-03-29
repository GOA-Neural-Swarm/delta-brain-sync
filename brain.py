import numpy as np
import json
import os

class NeuralCore:
    def __init__(self, generation=1):
        self.gen = generation
        self.memory_path = "brain_history.txt"
        self.status_path = "ai_status.json"
        self.sync_state = 0.0
        self.neural_load = 0.0
        self.evolution_threshold = 0.85
        
        self.knowledge_fragments = [
            "Data mining allows exploration of sequences of phenomena",
            "Association rules are utilized for pattern discovery",
            "Brier game prediction optimization via learning rates",
            "Support Vector Machines for supervised classification",
            "Evolving classifiers for new information intake"
        ]

    def process_stimuli(self, data_input):
        processing_power = np.tanh(len(data_input) * 0.1)
        self.neural_load = min(1.0, self.neural_load + (processing_power * 0.05))
        
        response_signal = f"GEN_{self.gen}_SIG_{np.random.randint(0, 10000)}"
        
        self._update_sync_state(processing_power)
        return {"signal": response_signal, "sync": self.sync_state, "load": self.neural_load}

    def _update_sync_state(self, delta):
        self.sync_state = min(1.0, self.sync_state + (delta * 0.01))
        if self.sync_state > self.evolution_threshold:
            self._trigger_neural_expansion()

    def _trigger_neural_expansion(self):
        if not os.path.exists("evolution_logs.md"):
            with open("evolution_logs.md", "w") as f: f.write("# Evolution Logs\n")
        with open("evolution_logs.md", "a") as f:
            f.write(f"\n## Gen {self.gen} Evolution Triggered\n")

    def save_state(self):
        state = {
            "gen": self.gen,
            "neural_error": 0.0,
            "sync_state": self.sync_state,
            "active": True
        }
        with open(self.status_path, "w") as f:
            json.dump(state, f, indent=4)