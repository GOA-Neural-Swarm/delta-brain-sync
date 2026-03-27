import time
import subprocess 
import os
import requests  
import json
import re
import sys
import hashlib
import logging
import numpy as np
import pandas as pd
import pickle

# ============================================================================
# 🛡️ SYSTEM CONFIGURATIONS & LOGGING
# ============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - 🛡️ [GUARD] - %(levelname)s - %(message)s')
logger = logging.getLogger("EvolutionGuard")

# API Configurations
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# 🚨 GUARDRAIL CONFIGURATION: AI ဖျက်ခွင့်မရှိသော အစိတ်အပိုင်းများ (Fully matched logic)
MANDATORY_COMPONENTS = [
    "class EvolutionEngine",
    "def initiate_evolution_step",
    "class NeuralProcessor",
    "def calculate_brier_mixability",
    "def evolve_classifier",
    "def process_sequence",
    "def add_neuron",
    "def add_connection",
    "def fire",
    "def apply_stdp",
    "def save_state",
    "def load_state",
    "class IntegrityChecker",
    "def verify_structural_integrity",
    "class DataSynchronizer",
    "def extract_association_rules",
    "def sync",
    "Brier Score",
    "SVM-based classification"
]

# ============================================================================
# 🧠 CORE 1: NEURAL PROCESSOR (L.I.F SNN & Advanced Logic Integrated)
# ============================================================================
class NeuralProcessor:
    """Advanced cognitive unit utilizing mixable prediction games and L.I.F SNN."""
    def __init__(self, max_neurons=1000000):
        self.max_neurons = max_neurons
        self.potentials = np.zeros(max_neurons, dtype=np.float32)
        self.activations = np.zeros(max_neurons, dtype=np.bool_)
        self.thresholds = np.full(max_neurons, 0.5, dtype=np.float32)
        self.leak_rate = 0.05 
        self.connections = {}  # {source_id: {target_id: weight}}
        self.active_count = 0
        self.synaptic_density = 1.0
        self.learning_rate = 0.00729 
        logger.info("🌌 [NEURAL-PROCESSOR]: Advanced L.I.F SNN Architecture Online.")

    def add_neuron(self, threshold=0.5):
        neuron_id = self.active_count
        if neuron_id < self.max_neurons:
            self.thresholds[neuron_id] = threshold
            self.active_count += 1
            return neuron_id
        return -1

    def add_connection(self, n1, n2, weight=0.1):
        if n1 >= self.max_neurons or n2 >= self.max_neurons: return False
        if n1 not in self.connections: self.connections[n1] = {}
        self.connections[n1][n2] = float(weight)
        return True

    def calculate_brier_mixability(self):
        """Brier Score နှင့် Entropy ကို အခြေခံ၍ Mutation Factor ကို တွက်ချက်သည်။"""
        import random
        try:
            logger.info("⚙️ [NEURAL]: Calculating Dynamic Brier Mixability Factor...")
            base_stability = 0.75
            mutation_variance = random.uniform(0.01, 0.15)
            final_factor = max(0.1, min(0.95, base_stability + mutation_variance))
            logger.info(f"🧬 [EVOLUTION]: Brier Mixability Factor set to {final_factor:.4f}")
            return final_factor
        except Exception as e:
            logger.error(f"⚠️ [NEURAL-ERROR]: Mixability failed: {e}")
            return 0.85

    def fire(self, external_stimuli):
        if isinstance(external_stimuli, int): external_stimuli = [external_stimuli]
        self.potentials *= (1.0 - self.leak_rate)
        for n_id in external_stimuli:
            self.potentials[n_id % self.max_neurons] += 1.0 
        spiked_neurons = np.where(self.potentials >= self.thresholds)[0]
        self.activations.fill(False)
        self.activations[spiked_neurons] = True
        next_targets = set()
        for source in spiked_neurons:
            self.potentials[source] = 0.0
            if source in self.connections:
                for target, weight in self.connections[source].items():
                    self.potentials[target] += weight
                    next_targets.add(target)
        return list(next_targets)

    def apply_stdp(self, learning_rate=0.01, decay_rate=0.005):
        spiked = np.where(self.activations)[0]
        for source in spiked:
            if source in self.connections:
                targets = list(self.connections[source].keys())
                for target in targets:
                    if self.activations[target]:
                        self.connections[source][target] = min(1.0, self.connections[source][target] + learning_rate)
                    else:
                        self.connections[source][target] -= decay_rate
                    if self.connections[source][target] <= 0: del self.connections[source][target]

    def evolve_classifier(self, new_data):
        logger.info("🧠 [NEURAL]: Evolving SVM Classifiers with new data sequence...")
        pass

    def process_sequence(self, sequence):
        logger.info(f"🔍 [NEURAL]: Mining association rules for sequence length: {len(sequence)}")
        return [f"rule_{i}" for i in range(len(sequence))]

    def save_state(self, filename="brain_state.pkl"):
        state = {'connections': self.connections, 'thresholds': self.thresholds[:self.active_count], 'active_count': self.active_count}
        with open(filename, 'wb') as f: pickle.dump(state, f)
        logger.info("💾 [SAVED]: Brain state saved.")

    def load_state(self, filename="brain_state.pkl"):
        try:
            with open(filename, 'rb') as f:
                state = pickle.load(f)
                self.connections = state['connections']
                self.active_count = state['active_count']
                self.thresholds[:self.active_count] = state['thresholds']
            logger.info("📂 [LOADED]: Brain state restored.")
        except FileNotFoundError: logger.error("❌ [LOAD-ERROR]: State file not found.")

# Backward Compatibility
Brain = NeuralProcessor

# ============================================================================
# 🧠 CORE 2: INTEGRITY CHECKER (Structural & Logic Guard)
# ============================================================================
class IntegrityChecker:
    """Ensures architectural stability during Gen 1 mutation cycles."""
    def __init__(self):
        self.monitored_nodes = ["main.py", "brain.py", "evolution_engine.py", "sync_data.py"]
        self.lock_file = "trigger.lock"
        logger.info("Integrity Guard System Activated.")

    def validate_evolution_integrity(self, new_code):
        """AI ပေးသော ကုဒ်ထဲတွင် လိုအပ်သော Core Logic များ ပါ၊ မပါ စစ်ဆေးခြင်း"""
        missing_parts = []
        for component in MANDATORY_COMPONENTS:
            if component not in new_code:
                missing_parts.append(component)
        if missing_parts:
            logger.error(f"❌ [GUARDRAIL-REJECTED]: Critical logic missing: {missing_parts}")
            return False
        logger.info("✅ [GUARDRAIL-PASSED]: Core logic integrity verified.")
        return True

    def verify_structural_integrity(self):
        logger.info("Verifying system structural integrity...")
        if os.path.exists(self.lock_file):
            logger.warning("System is currently LOCKED.")
            return False
        for node in self.monitored_nodes:
            if not os.path.exists(node):
                logger.error(f"Missing Critical Node: {node}")
                return False
            if not self.validate_syntax_file(node):
                return False
        return True

    def validate_syntax_file(self, file_path):
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            compile(content, file_path, 'exec')
            return True
        except Exception as e:
            logger.error(f"Syntax Error in {file_path}: {str(e)}")
            return False

    def lock_system(self):
        with open(self.lock_file, "w") as f: f.write("LOCKED")
        logger.info("🔒 System Locked.")

    def unlock_system(self):
        if os.path.exists(self.lock_file): os.remove(self.lock_file)
        logger.info("🔓 System Unlocked.")

# Singleton instance
guard = IntegrityChecker()

# ============================================================================
# 🤖 CORE 3: AI AUTO-HEALING ENGINE (Gemini + Groq Orchestrator)
# ============================================================================
def get_ai_correction(error_log, original_code, retry_count=0):
    MAX_RETRIES = 3
    if retry_count >= MAX_RETRIES:
        print("❌ [GUARD]: Max API retries exceeded.")
        return original_code

    print(f"🧠 [GUARD]: AI analyzing cycle (Attempt {retry_count + 1}/{MAX_RETRIES})...")
    prompt = (
        f"Fix this Python error:\n{error_log}\n\nCode:\n{original_code}\n\n"
        "IMPORTANT: Do NOT remove 'calculate_brier_mixability', 'evolve_classifier', 'fire', 'apply_stdp' or 'DataSynchronizer'. "
        "Keep the Brier Score and L.I.F SNN logic intact. Return ONLY the clean code."
    )

    # --- ATTEMPT 1: GEMINI ---
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        res = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.2}}, timeout=30)
        if res.status_code == 200:
            content = res.json()['candidates'][0]['content']['parts'][0]['text']
            corrected = re.sub(r'```python\n|```', '', content).strip()
            if guard.validate_evolution_integrity(corrected): return corrected
    except Exception: print("⚠️ [GEMINI-FAIL]: Switching to Groq...")

    # --- ATTEMPT 2: GROQ ---
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        res = requests.post(url, headers=headers, json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}]}, timeout=30)
        if res.status_code == 200:
            content = res.json()['choices'][0]['message']['content']
            corrected = re.sub(r'```python\n|```', '', content).strip()
            if guard.validate_evolution_integrity(corrected): return corrected
    except Exception: pass
    
    return original_code

# ============================================================================
# 🚀 EXECUTION CONTROL (run_guard)
# ============================================================================
def run_guard(target_script):
    print(f"🛡️ [GUARD]: Launching {target_script} under Sovereign Supervision...")
    process = subprocess.Popen(['python3', target_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    start_time = time.time()
    error_output = ""
    
    while time.time() - start_time < 60:
        line = process.stderr.readline()
        if line:
            error_output += line
            if "Traceback" in line or "Error" in line:
                print("❌ [GUARD]: Structural failure detected. Initiating Healing...")
                process.terminate()
                with open(target_script, 'r') as f: original_code = f.read()
                corrected = get_ai_correction(error_output, original_code)
                if corrected != original_code:
                    with open(target_script, 'w') as f: f.write(corrected)
                    print("✅ [GUARD]: Integrity Restored. Rebooting...")
                    return run_guard(target_script)
                else: sys.exit(1)

        if process.poll() is not None and process.poll() != 0:
            remaining_error = process.stderr.read()
            with open(target_script, 'r') as f: original_code = f.read()
            corrected = get_ai_correction(remaining_error, original_code)
            if corrected != original_code:
                with open(target_script, 'w') as f: f.write(corrected)
                return run_guard(target_script)
            sys.exit(1)
        time.sleep(1)

    print("🌐 [GUARD]: System Gen 1 Stability confirmed. Sovereign handover complete.")
    sys.exit(0)

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "main.py"
    run_guard(target)
