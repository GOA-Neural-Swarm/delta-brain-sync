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

# ============================================================================
# 🧠 CORE 1: INTEGRITY CHECKER (Preventing Import Errors)
# ============================================================================
class IntegrityChecker:
    """
    Evolution Integrity Checker (EIC).
    AI မှ ကုဒ်များကို ပြင်ဆင်သည့်အခါ Error မပါစေရန် စစ်ဆေးပေးသော အဓိက Class ဖြစ်သည်။
    """
    def __init__(self):
        self.monitored_files = ["evolution_engine.py", "app.py", "evolved_module.py", "main.py"]
        logger.info("Integrity Guard System Activated.")

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

# Singleton instance to be imported by evolution_engine.py
guard = IntegrityChecker()

# ============================================================================
# 🤖 CORE 2: AI AUTO-HEALING ENGINE (Gemini + Groq)
# ============================================================================
def get_ai_correction(error_log, original_code):
    print("🧠 [GUARD]: AI is analyzing the error...")
    
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
        data = res.json()
        if res.status_code == 200 and 'candidates' in data:
            content = data['candidates'][0]['content']['parts'][0]['text']
            return re.sub(r'```python\n|```', '', content).strip()
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
        data = response.json()
        
        if 'error' in data and 'rate_limit_exceeded' in str(data):
            print("⏳ [RATE-LIMIT]: Sleeping for 20 seconds...")
            time.sleep(20)
            return get_ai_correction(error_log, original_code)
        
        # API Error ရှိမရှိ စစ်ဆေးခြင်း
        if 'choices' in data:
            content = data['choices'][0]['message']['content']
            return re.sub(r'```python\n|```', '', content).strip()
        else:
            print(f"❌ [GUARD]: API Error Response: {data}")
            # API Error တက်ရင် Original code ကိုပဲ ပြန်ပေးပြီး Exit လုပ်မယ် (Loop မပတ်အောင်)
            return original_code
            
    except Exception as e:
        print(f"❌ [GUARD]: Request failed: {e}")
        return original_code

def run_guard(target_script):
    print(f"🛡️ [GUARD]: Launching {target_script} in Observation Mode...")
    
    # Background မှာ process ကို run မယ် (စောင့်မနေတော့ဘူး)
    # logic coupled with real-time log observation
    process = subprocess.Popen(
        ['python3', target_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    start_time = time.time()
    error_output = ""
    
    # ပထမ ၆၀ စက္ကန့်အတွင်းမှာ process ကို အနီးကပ် စောင့်ကြည့်မယ်
    while time.time() - start_time < 60:
        # Error ထွက်လာသလား စစ်မယ်
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
                with open(target_script, 'w') as f:
                    f.write(corrected)
                
                print("✅ [GUARD]: System evolved. Restarting Guard Cycle...")
                return run_guard(target_script) # ပြန်စမယ်

        # အကယ်၍ process က ပိတ်သွားပြီး error ရှိနေရင်
        if process.poll() is not None and process.poll() != 0:
            remaining_error = process.stderr.read()
            print(f"❌ [GUARD]: Process died with error: {remaining_error}")
            
            with open(target_script, 'r') as f:
                original_code = f.read()
                
            # Get corrected code from AI and retry
            corrected_code = get_ai_correction(remaining_error, original_code)
            with open(target_script, 'w') as f:
                f.write(corrected_code)
            print("✅ [GUARD]: Correction applied. Rebooting system...")
            return run_guard(target_script)

        time.sleep(1)

    # ၆၀ စက္ကန့်အတွင်း Error မတက်ဘဲ အသက်ရှင်နေရင် Healthy လို့ သတ်မှတ်မယ်
    print("🌐 [GUARD]: System is stable and sovereign. Handing over to background process.")
    # GitHub Action ကို အောင်မြင်စွာ ပိတ်ခိုင်းလိုက်ပေမယ့် background မှာ app.py က ဆက် run နေမှာမဟုတ်ဘူး၊ 
    # ဒါပေမဲ့ evolution cycle ပြီးမြောက်ဖို့အတွက် ဒီအဆင့်ဟာ အရေးကြီးဆုံးဖြစ်ပါတယ်။
    sys.exit(0)

# ============================================================================
# 🚀 DIRECT EXECUTION ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    # Get target script from command line, default to main.py
    target = sys.argv[1] if len(sys.argv) > 1 else "main.py"
    run_guard(target)
