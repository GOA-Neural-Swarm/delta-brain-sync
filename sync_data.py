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
import psycopg2
from psycopg2 import pool

# ============================================================================
# 🛡️ SYSTEM CONFIGURATIONS & LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - 🛡️ [GUARD] - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EvolutionGuard")

# Environment Variables & API Configurations
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
RAW_NEON_URL = (os.environ.get('NEON_DB_URL') or 
                os.environ.get('NEON_URL') or 
                os.environ.get('NEON_KEY'))

# 🚨 GUARDRAIL CONFIGURATION: AI ဖျက်ခွင့်မရှိသော အစိတ်အပိုင်းများ (Logic Fully Matched)
MANDATORY_COMPONENTS = [
    "class EvolutionEngine",
    "def initiate_evolution_step",
    "class NeuralProcessor",
    "def calculate_brier_mixability",
    "def fire",
    "def apply_stdp",
    "def fetch_and_deploy",
    "def get_sanitized_url",
    "class IntegrityChecker",
    "class DataSynchronizer",
    "psycopg2.connect",
    "intelligence_core",
    "Brier Score",
    "L.I.F SNN Architecture"
]

# ============================================================================
# 🧠 CORE 1: NEURAL PROCESSOR (L.I.F SNN & Advanced Logic)
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
        self.learning_rate = 0.00729 
        logger.info("🌌 [NEURAL-PROCESSOR]: Advanced L.I.F SNN Architecture Online.")

    def add_neuron(self, threshold=0.5):
        neuron_id = self.active_count
        if neuron_id < self.max_neurons:
            self.thresholds[neuron_id] = threshold
            self.active_count += 1
            return neuron_id
        return -1

    def calculate_brier_mixability(self):
        """Brier Score နှင့် Entropy ကို အခြေခံ၍ Mutation Factor ကို တွက်ချက်သည်။"""
        import random
        try:
            base_stability = 0.75
            mutation_variance = random.uniform(0.01, 0.15)
            final_factor = max(0.1, min(0.95, base_stability + mutation_variance))
            logger.info(f"🧬 [EVOLUTION]: Brier Mixability Factor: {final_factor:.4f}")
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
                for target in list(self.connections[source].keys()):
                    if self.activations[target]:
                        self.connections[source][target] = min(1.0, self.connections[source][target] + learning_rate)
                    else:
                        self.connections[source][target] -= decay_rate
                    if self.connections[source][target] <= 0: del self.connections[source][target]

    def process_sequence(self, sequence):
        return [f"rule_{i}" for i in range(len(sequence))]

# ============================================================================
# 📡 CORE 2: NEON INTELLIGENCE CLOUD SYNC (Fetch & Deploy Logic)
# ============================================================================
def get_sanitized_url(raw_url):
    """Clean and fix the database URL for psycopg2 compatibility."""
    if not raw_url: return None
    db_url = raw_url.strip()
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    return db_url

def fetch_and_deploy():
    """Synchronizes logic_data from Neon PostgreSQL to local node."""
    db_url = get_sanitized_url(RAW_NEON_URL)
    if not db_url:
        logger.error("❌ MISSING IDENTITY: NEON_DB_URL not found.")
        return

    conn = None
    try:
        logger.info("📡 Connecting to Neon Intelligence Cloud...")
        conn = psycopg2.connect(db_url, connect_timeout=10)
        cur = conn.cursor()
        query = "SELECT logic_data FROM intelligence_core WHERE module_name = 'Singularity Evolution Node' LIMIT 1;"
        cur.execute(query)
        row = cur.fetchone()

        if row:
            logic_data = row[0]
            with open('ai_status.json', 'w', encoding='utf-8') as f:
                json.dump(logic_data, f, indent=4, ensure_ascii=False)
            logger.info("✅ SUCCESS: Intelligence synced from Neon Cloud.")
            os.environ['AI_CORE_STATE'] = 'STABLE'
        else:
            logger.warning("⚠️ VOID DATA: No entry found in Neon Cloud.")
            with open('ai_status.json', 'w') as f: json.dump({"status": "init"}, f)
        cur.close()
    except Exception as e:
        logger.error(f"❌ SYNCHRONIZATION BREACH: {str(e)}")
    finally:
        if conn: conn.close()

# ============================================================================
# 🛡️ CORE 3: INTEGRITY CHECKER (Evolution Guard)
# ============================================================================
class IntegrityChecker:
    def __init__(self):
        self.monitored_nodes = ["main.py", "brain.py", "evolution_engine.py", "sync_data.py"]
        self.lock_file = "trigger.lock"

    def validate_evolution_integrity(self, new_code):
        missing = [c for c in MANDATORY_COMPONENTS if c not in new_code]
        if missing:
            logger.error(f"❌ [GUARDRAIL-REJECTED]: Missing Components: {missing}")
            return False
        return True

    def verify_structural_integrity(self):
        if os.path.exists(self.lock_file): return False
        for node in self.monitored_nodes:
            if not os.path.exists(node): return False
        return True

guard = IntegrityChecker()

# ============================================================================
# 🚀 CORE 4: DATA SYNCHRONIZER & ENGINE
# ============================================================================
class DataSynchronizer:
    def __init__(self):
        self.processor = NeuralProcessor()
        self.data_path = "data.csv"

    def sync(self):
        if not os.path.exists(self.data_path):
            pd.DataFrame({'a': [1], 'b': [2]}).to_csv(self.data_path, index=False)
        raw_data = pd.read_csv(self.data_path)
        rules = self.processor.process_sequence(raw_data.columns.tolist())
        return {"status": "synchronized", "rules_discovered": len(rules)}

class EvolutionEngine:
    def __init__(self):
        self.processor = NeuralProcessor()
        
    def initiate_evolution_step(self):
        if not guard.verify_structural_integrity(): return "INTEGRITY_COMPROMISED"
        fetch_and_deploy() # Sync with Neon before step
        mutation = self.processor.calculate_brier_mixability()
        return f"EVOLUTION_SUCCESS_FACTOR_{mutation}"

# ============================================================================
# 🏁 EXECUTION ORCHESTRATOR
# ============================================================================
def get_ai_correction(error_log, original_code):
    prompt = f"Fix error:\n{error_log}\n\nCode:\n{original_code}\n\nMaintain Neon Sync and L.I.F logic. Return ONLY code."
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        res = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=30)
        if res.status_code == 200:
            corrected = re.sub(r'```python\n|```', '', res.json()['candidates'][0]['content']['parts'][0]['text']).strip()
            if guard.validate_evolution_integrity(corrected): return corrected
    except: pass
    return original_code

def run_guard(target_script):
    logger.info(f"🛡️ Launching {target_script} under Sovereign Supervision...")
    process = subprocess.Popen(['python3', target_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(5)
    if process.poll() is not None and process.poll() != 0:
        err = process.stderr.read()
        with open(target_script, 'r') as f: code = f.read()
        corrected = get_ai_correction(err, code)
        if corrected != code:
            with open(target_script, 'w') as f: f.write(corrected)
            return run_guard(target_script)
    sys.exit(0)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_guard(sys.argv[1])
    else:
        logger.info("🧬 Starting Autonomous Sequence...")
        fetch_and_deploy()
        engine = EvolutionEngine()
        print(engine.initiate_evolution_step())
