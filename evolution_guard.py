import time
import subprocess 
import os
import requests  
import json
import re
import sys
import hashlib
import logging

# ============================================================================
# 🛡️ SYSTEM CONFIGURATIONS & LOGGING
# ============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - 🛡️ [GUARD] - %(levelname)s - %(message)s')
logger = logging.getLogger("EvolutionGuard")

# API Configurations
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# 🚨 GUARDRAIL CONFIGURATION: AI ဖျက်ခွင့်မရှိသော အစိတ်အပိုင်းများ
MANDATORY_COMPONENTS = [
    "class NeuralProcessor",
    "def calculate_brier_mixability",
    "class EvolutionEngine",
    "def initiate_evolution_step"
]

# ============================================================================
# 🧠 CORE 1: INTEGRITY CHECKER (Preventing Import Errors & Guardrail Enforcement)
# ============================================================================
class IntegrityChecker:
    """
    Evolution Integrity Checker (EIC).
    AI မှ ကုဒ်များကို ပြင်ဆင်သည့်အခါ Error မပါစေရန်နှင့် Core Logic များ မပျောက်ပျက်စေရန် စစ်ဆေးပေးသည်။
    """
    def __init__(self):
        self.monitored_files = ["evolution_engine.py", "app.py", "evolved_module.py", "main.py"]
        logger.info("Integrity Guard System Activated.")

    def validate_evolution_integrity(self, new_code):
        """AI ပေးသော ကုဒ်ထဲတွင် လိုအပ်သော Core Logic များ ပါ၊ မပါ စစ်ဆေးခြင်း"""
        missing_parts = []
        for component in MANDATORY_COMPONENTS:
            if component not in new_code:
                missing_parts.append(component)
        
        if missing_parts:
            logger.error(f"❌ [GUARDRAIL-REJECTED]: AI tried to remove critical logic: {missing_parts}")
            return False
        
        logger.info("✅ [GUARDRAIL-PASSED]: Core logic integrity verified.")
        return True

    def verify_structural_integrity(self):
        """Engine မှ တောင်းဆိုနေသော အဓိက Function"""
        logger.info("Verifying system structural integrity...")
        for file in self.monitored_files:
            if os.path.exists(file):
                if not self.validate_syntax_file(file):
                    return False
        return True

    def validate_syntax_file(self, file_path):
        """ဖိုင်တစ်ခုချင်းစီ၏ Syntax ကို စစ်ဆေးရန်"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            compile(content, file_path, 'exec')
            return True
        except Exception as e:
            logger.error(f"Syntax Error in {file_path}: {str(e)}")
            return False
    
    def check_file_health(self, file_path):
        if not os.path.exists(file_path):
            logger.error(f"Missing Critical File: {file_path}")
            return False
        return True

    def validate_syntax(self, code_string):
        try:
            compile(code_string, '<string>', 'exec')
            return True
        except SyntaxError as e:
            logger.error(f"Syntax Validation Failed: {str(e)}")
            return False

    def get_file_hash(self, file_path):
        hasher = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                buf = f.read()
                hasher.update(buf)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Hashing Error: {str(e)}")
            return None

# Singleton instance to be used throughout the script
guard = IntegrityChecker()

# ============================================================================
# 🤖 CORE 2: AI AUTO-HEALING ENGINE (Gemini + Groq with Guardrail Protection)
# ============================================================================
def get_ai_correction(error_log, original_code, retry_count=0):
    MAX_RETRIES = 3
    if retry_count >= MAX_RETRIES:
        print("❌ [GUARD]: Max API retries exceeded. Aborting to prevent infinite loop.")
        return original_code

    print(f"🧠 [GUARD]: AI is analyzing the error (Attempt {retry_count + 1}/{MAX_RETRIES})...")
    
    prompt = f"Fix this Python error:\n{error_log}\n\nCode:\n{original_code}\n\nReturn ONLY the clean code."

    # --- ATTEMPT 1: GEMINI (Primary) ---
    print("📡 [GUARD-GEMINI]: Requesting correction...")
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    gemini_payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2}
    }

    try:
        res = requests.post(gemini_url, json=gemini_payload, timeout=30)
        
        # Explicit Gemini Rate Limit Handling (Status 429)
        if res.status_code == 429:
            backoff_time = 20 * (2 ** retry_count) # Exponential backoff: 20s, 40s, 80s
            print(f"⏳ [GEMINI-RATE-LIMIT]: Status 429 detected. Sleeping for {backoff_time}s...")
            time.sleep(backoff_time)
            return get_ai_correction(error_log, original_code, retry_count + 1)

        data = res.json()
        if res.status_code == 200 and 'candidates' in data:
            content = data['candidates'][0]['content']['parts'][0]['text']
            corrected_code = re.sub(r'```python\n|```', '', content).strip()
            
            # 🚨 [NEW] Guardrail Integrity Check
            if guard.validate_evolution_integrity(corrected_code):
                return corrected_code
            else:
                print("⚠️ [GUARDRAIL]: Output rejected. Re-attempting AI generation...")
                return get_ai_correction(error_log, original_code, retry_count + 1)
        else:
            print(f"⚠️ [GEMINI-FAIL]: Status {res.status_code}. Switching to Groq...")
    except Exception as e:
        print(f"⚠️ [GEMINI-ERROR]: {e}. Switching to Groq...")

    # --- ATTEMPT 2: GROQ (Fallback) ---
    print("📡 [GUARD-GROQ]: Requesting fallback correction...")
    groq_url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    
    groq_payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}]
    }
    
    try:
        response = requests.post(groq_url, headers=headers, json=groq_payload, timeout=30)
        
        # Explicit Groq Rate Limit Handling (Status 429)
        if response.status_code == 429:
            backoff_time = 20 * (2 ** retry_count) 
            print(f"⏳ [GROQ-RATE-LIMIT]: Status 429 detected. Sleeping for {backoff_time}s...")
            time.sleep(backoff_time)
            return get_ai_correction(error_log, original_code, retry_count + 1)

        data = response.json()
        
        # Additional JSON body check for Groq specific rate limit messages
        if 'error' in data and 'rate_limit_exceeded' in str(data):
            backoff_time = 20 * (2 ** retry_count)
            print(f"⏳ [GROQ-API-LIMIT]: Sleeping for {backoff_time} seconds...")
            time.sleep(backoff_time)
            return get_ai_correction(error_log, original_code, retry_count + 1)
        
        if 'choices' in data:
            content = data['choices'][0]['message']['content']
            corrected_code = re.sub(r'```python\n|```', '', content).strip()
            
            # 🚨 [NEW] Guardrail Integrity Check for Groq
            if guard.validate_evolution_integrity(corrected_code):
                return corrected_code
            else:
                return original_code # Do not save broken code if fallback also fails guardrail
        else:
            print(f"❌ [GUARD]: API Error Response: {data}")
            return original_code
            
    except Exception as e:
        print(f"❌ [GUARD]: Request failed: {e}")
        return original_code

def run_guard(target_script):
    print(f"🛡️ [GUARD]: Launching {target_script} in Observation Mode...")
    
    process = subprocess.Popen(
        ['python3', target_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    start_time = time.time()
    error_output = ""
    
    while time.time() - start_time < 60:
        line = process.stderr.readline()
        if line:
            print(f"⚠️ [LOG]: {line.strip()}")
            error_output += line
            if "Traceback" in line or "Error" in line:
                print("❌ [GUARD]: Critical Error detected! Terminating and fixing...")
                process.terminate()
                
                with open(target_script, 'r') as f:
                    original_code = f.read()
                
                corrected = get_ai_correction(error_output, original_code)
                
                if corrected != original_code:
                    with open(target_script, 'w') as f:
                        f.write(corrected)
                    print("✅ [GUARD]: System evolved. Restarting Guard Cycle...")
                    return run_guard(target_script) 
                else:
                    print("⚠️ [GUARD]: AI could not fix the code or max retries reached. Guard stopping.")
                    sys.exit(1)

        if process.poll() is not None and process.poll() != 0:
            remaining_error = process.stderr.read()
            print(f"❌ [GUARD]: Process died with error: {remaining_error}")
            
            with open(target_script, 'r') as f:
                original_code = f.read()
                
            corrected_code = get_ai_correction(remaining_error, original_code)
            
            if corrected_code != original_code:
                with open(target_script, 'w') as f:
                    f.write(corrected_code)
                print("✅ [GUARD]: Correction applied. Rebooting system...")
                return run_guard(target_script)
            else:
                 print("⚠️ [GUARD]: AI could not fix the code or max retries reached. Guard stopping.")
                 sys.exit(1)

        time.sleep(1)

    print("🌐 [GUARD]: System is stable and sovereign. Handing over to background process.")
    sys.exit(0)

# ============================================================================
# 🚀 DIRECT EXECUTION ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "main.py"
    run_guard(target)
