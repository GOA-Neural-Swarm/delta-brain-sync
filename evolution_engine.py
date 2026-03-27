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

# ============================================================================
# 🛡️ SYSTEM CONFIGURATIONS & LOGGING
# ============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - 🛡️ [GUARD] - %(levelname)s - %(message)s')
logger = logging.getLogger("EvolutionGuard")

# API Configurations
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# 🚨 GUARDRAIL CONFIGURATION: AI ဖျက်ခွင့်မရှိသော အစိတ်အပိုင်းများ (Fully matched from provided snippets)
MANDATORY_COMPONENTS = [
    "class EvolutionEngine",
    "def initiate_evolution_step",
    "class NeuralProcessor",
    "def calculate_brier_mixability",
    "def evolve_classifier",
    "def process_sequence",
    "class IntegrityChecker",
    "def verify_structural_integrity",
    "class DataSynchronizer",
    "def extract_association_rules",
    "def sync",
    "Brier Score",
    "SVM-based classification"
]

# ============================================================================
# 🧠 CORE 1: INTEGRITY CHECKER (Preserving snippets logic)
# ============================================================================
class IntegrityChecker:
    """Ensures architectural stability during Gen 1 mutation cycles."""
    def __init__(self):
        # Monitored nodes from snippets + core system files
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
        """Ensures architectural stability during mutation cycles."""
        logger.info("Verifying system structural integrity...")
        if os.path.exists(self.lock_file):
            logger.warning("System is currently LOCKED.")
            return False
            
        for node in self.monitored_nodes:
            if not os.path.exists(node):
                logger.error(f"Missing Critical Node: {node}")
                return False
            # Syntax validation check
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
        with open(self.lock_file, "w") as f:
            f.write("LOCKED")
        logger.info("🔒 System Locked.")

    def unlock_system(self):
        if os.path.exists(self.lock_file):
            os.remove(self.lock_file)
            logger.info("🔓 System Unlocked.")

# Singleton instance
guard = IntegrityChecker()

# ============================================================================
# 🤖 CORE 2: AI AUTO-HEALING ENGINE (Gemini + Groq with Guardrail Integration)
# ============================================================================
def get_ai_correction(error_log, original_code, retry_count=0):
    MAX_RETRIES = 3
    if retry_count >= MAX_RETRIES:
        print("❌ [GUARD]: Max API retries exceeded.")
        return original_code

    print(f"🧠 [GUARD]: AI analyzing cycle (Attempt {retry_count + 1}/{MAX_RETRIES})...")
    
    # AI prompts emphasize keeping Brier mixability and SVM logic
    prompt = (
        f"Fix this Python error:\n{error_log}\n\n"
        f"Code:\n{original_code}\n\n"
        "IMPORTANT: Do NOT remove 'calculate_brier_mixability', 'evolve_classifier', or 'DataSynchronizer'. "
        "Keep the Brier Score calculation logic intact. Return ONLY the clean code."
    )

    # --- ATTEMPT 1: GEMINI ---
    try:
        gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        res = requests.post(gemini_url, json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.2}}, timeout=30)
        
        if res.status_code == 429:
            time.sleep(20 * (2 ** retry_count))
            return get_ai_correction(error_log, original_code, retry_count + 1)

        data = res.json()
        if res.status_code == 200 and 'candidates' in data:
            content = data['candidates'][0]['content']['parts'][0]['text']
            corrected_code = re.sub(r'```python\n|```', '', content).strip()
            
            if guard.validate_evolution_integrity(corrected_code):
                return corrected_code
            else:
                return get_ai_correction(error_log, original_code, retry_count + 1)
    except Exception:
        print("⚠️ [GEMINI-FAIL]: Switching to Groq...")

    # --- ATTEMPT 2: GROQ ---
    try:
        groq_url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        response = requests.post(groq_url, headers=headers, json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}]}, timeout=30)
        
        if response.status_code == 429:
            time.sleep(20 * (2 ** retry_count))
            return get_ai_correction(error_log, original_code, retry_count + 1)

        data = response.json()
        if 'choices' in data:
            content = data['choices'][0]['message']['content']
            corrected_code = re.sub(r'```python\n|```', '', content).strip()
            
            if guard.validate_evolution_integrity(corrected_code):
                return corrected_code
    except Exception:
        return original_code
    
    return original_code

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
                
                with open(target_script, 'r') as f:
                    original_code = f.read()
                
                corrected = get_ai_correction(error_output, original_code)
                
                if corrected != original_code:
                    with open(target_script, 'w') as f:
                        f.write(corrected)
                    print("✅ [GUARD]: Integrity Restored. Rebooting...")
                    return run_guard(target_script)
                else:
                    sys.exit(1)

        if process.poll() is not None and process.poll() != 0:
            remaining_error = process.stderr.read()
            with open(target_script, 'r') as f:
                original_code = f.read()
            corrected = get_ai_correction(remaining_error, original_code)
            if corrected != original_code:
                with open(target_script, 'w') as f:
                    f.write(corrected)
                return run_guard(target_script)
            sys.exit(1)
        time.sleep(1)

    print("🌐 [GUARD]: System Gen 1 Stability confirmed. Sovereign handover complete.")
    sys.exit(0)

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "main.py"
    run_guard(target)
