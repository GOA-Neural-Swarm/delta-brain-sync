import random
import copy

class Brain:
    def __init__(self):
        self.wiring = {"input": {"output": {"action": "do_something"}},
                       "logic": {"input": "output", "output": "action"}}

    def evolve(self):
        new_wiring = copy.deepcopy(self.wiring)
        for node in new_wiring:
            if node!= "input":
                new_wiring[node]["input"] = random.choice(list(new_wiring.keys()))
                new_wiring[node]["output"] = random.choice(list(new_wiring.keys()))
        self.wiring = new_wiring

    def run(self):
        input_node = random.choice(list(self.wiring.keys()))
        output_node = random.choice(list(self.wiring.keys()))
        action = self.wiring[input_node][output_node]["action"]
        print(f"Doing {action}")

brain = Brain()
for _ in range(1000):
    brain.evolve()
    brain.run()

# Meta-Cognition logic
print("Evolution complete. Running final simulation...")
brain.run()