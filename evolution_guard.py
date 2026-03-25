import subprocess
import os
import requests 
import json
import re
import sys
import time

# API Configurations
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

def get_ai_correction(error_log, original_code):
    print("🧠 [GUARD]: AI is analyzing the error...")
    
    prompt = f"Fix this Python error:\n{error_log}\n\nCode:\n{original_code}\n\nReturn ONLY the clean code."

    # --- ATTEMPT 1: GEMINI (Primary) ---
    print("📡 [GUARD-GEMINI]: Requesting correction...")
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={GEMINI_API_KEY}"
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
    # 1. Run the target script
    print(f"🛡️ [GUARD]: Executing {target_script}...")
    result = subprocess.run(['python', target_script], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ [GUARD]: {target_script} failed. Analyzing...")
        
        with open(target_script, 'r') as f:
            original_code = f.read()
            
        # 2. Get corrected code from AI
        corrected_code = get_ai_correction(result.stderr, original_code)
        
        # 3. Apply the correction
        with open(target_script, 'w') as f:
            f.write(corrected_code)
        print("✅ [GUARD]: Correction applied. Rebooting system...")
        
        # 4. Retry
        run_guard(target_script)
    else:
        print("✅ [GUARD]: System is stable.")

if __name__ == "__main__":
    # Get target script from command line, default to main.py
    target = sys.argv[1] if len(sys.argv) > 1 else "main.py"
    run_guard(target)
