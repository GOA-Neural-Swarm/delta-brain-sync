import numpy as np
import requests
import base64
import time
from kaggle_secrets import UserSecretsClient

# --- SECTION 1: YOUR ORIGINAL BRAIN LOGIC (UNTOUCHED) ---
class Brain:
    def __init__(self):
        self.memory = np.random.rand(1000)  # Initialize memory array
        self.connections = {}  # Initialize connections dictionary

    def learn(self, input_data, output_data):
        # Calculate error
        error = np.mean((output_data - self.memory) ** 2)

        # Update memory
        self.memory = np.add(self.memory, error * (input_data - self.memory))

        # Update connections
        for i in range(len(self.memory)):
            if self.memory[i] > 0.5:
                self.connections[i] = np.random.rand()

    def think(self, input_data):
        output_data = np.zeros((1000,))
        for i in range(len(input_data)):
            output_data += self.memory * input_data[i]
        return output_data

# --- SECTION 2: AUTO-EVOLUTION ENGINE ---
def evolve_and_sync(gen_count):
    # Initialize brain & Train (Your Logic)
    brain = Brain()
    for i in range(100): # Training cycle
        input_data = np.random.rand(1000)
        output_data = np.random.rand(1000)
        brain.learn(input_data, output_data)

    # GitHub Sync Config
    user_secrets = UserSecretsClient()
    try:
        TOKEN = user_secrets.get_secret("GH_TOKEN")
        REPO = "GOA-Neural-Swarm/delta-brain-sync"
        FILE_PATH = "brain.py"
        
        url = f"https://api.github.com/repos/{REPO}/contents/{FILE_PATH}"
        headers = {"Authorization": f"token {TOKEN}"}
        
        # Get SHA to update existing file
        r = requests.get(url, headers=headers)
        sha = r.json().get('sha') if r.status_code == 200 else None
        
        # Self-Refining Code Injection
        new_gen = gen_count + 1
        with open(__file__, 'r') as f:
            current_code = f.read()
            
        header_comment = f"# --- [ PREDATOR GEN {new_gen} ACTIVE ] ---\n"
        full_content = header_comment + current_code
        
        encoded = base64.b64encode(full_content.encode()).decode()
        data = {
            "message": f"🔱 Gen {new_gen} Evolution Pulse",
            "content": encoded,
            "sha": sha
        }
        
        res = requests.put(url, headers=headers, json=data)
        if res.status_code in [200, 201]:
            print(f"✅ [SUCCESS]: Gen {new_gen} Synchronized to GitHub.")
            return new_gen
        else:
            print(f"❌ [SYNC ERROR]: {res.status_code}")
            return gen_count
            
    except Exception as e:
        print(f"⚠️ [SYSTEM CRITICAL]: {e}")
        return gen_count

# --- SECTION 3: EXECUTION LOOP ---
if __name__ == "__main__":
    # Test initial brain (Your Logic)
    test_brain = Brain()
    test_input = np.random.rand(1000)
    print("🧠 Brain Output Test:", test_brain.think(test_input)[:5], "...") # Show first 5 values

    # Infinite Evolution Loop
    current_gen = 6200 # Starting point
    while True:
        current_gen = evolve_and_sync(current_gen)
        time.sleep(60) # Sync every 1 minute
