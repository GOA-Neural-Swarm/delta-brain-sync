import os
import subprocess
import requests
import json
import re
import sys
import shutil

# API Configurations
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

class EvolutionEngine:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.api_key}"
        self.max_retries = 3 # တစ်ဖိုင်ကို အများဆုံး ၃ ကြိမ်ပဲ ကြိုးစားပြင်မယ်

    def ask_ai_to_fix(self, error_log, file_content, file_name):
        print(f"🧠 [BRAIN]: Analyzing issues in {file_name}...")
        
        prompt = f"""
        Role: Senior Python Engineer & System Architect.
        Task: Fix Deprecation Warnings and Runtime Errors.
        
        Input Error/Warning:
        {error_log}
        
        Target File: {file_name}
        Current Code:
        {file_content}
        
        Requirements:
        1. Fix all warnings and errors mentioned.
        2. Upgrade all deprecated library calls to their latest versions (e.g., use 'google-genai' instead of 'google-generativeai').
        3. Ensure the code remains functionally identical but modern.
        4. Return ONLY the complete corrected Python code inside a code block.
        """
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1} # တိကျမှုရှိအောင် low temperature ထားမယ်
        }
        
        try:
            res = requests.post(self.url, json=payload, timeout=90)
            res.raise_for_status()
            data = res.json()
            raw_output = data['candidates'][0]['content']['parts'][0]['text']
            
            # Markdown code block တွေကို ဖယ်ရှားခြင်း
            clean_code = re.sub(r'```python\s*|```', '', raw_output).strip()
            return clean_code
        except Exception as e:
            print(f"❌ [AI-ERROR]: Analysis failed: {e}")
            return None

    def run_and_evolve(self, target_file):
        if not os.path.exists(target_file):
            return

        for attempt in range(self.max_retries):
            print(f"🔍 [SCAN]: Checking {target_file} (Attempt {attempt + 1})...")
            
            # Subprocess ကိုသုံးပြီး Warning ရော Error ရော ဖမ်းမယ်
            # PYTHONWARNINGS=default က warning တွေကို stderr ထဲ ပို့ပေးတယ်
            env = os.environ.copy()
            env["PYTHONWARNINGS"] = "default"
            
            process = subprocess.run(
                [sys.executable, target_file],
                capture_output=True,
                text=True,
                env=env
            )

            # Warning သို့မဟုတ် Error ရှိမရှိ စစ်မယ်
            issue_log = process.stderr.strip()
            
            if not issue_log:
                print(f"✅ [STABLE]: No issues detected in {target_file}.")
                break
            
            print(f"⚠️ [ISSUE-FOUND]: Detected problem in {target_file}.")
            
            with open(target_file, 'r') as f:
                original_code = f.read()

            # Backup လုပ်ထားခြင်း (Safety First)
            shutil.copy(target_file, f"{target_file}.bak")

            # AI ဆီက အဖြေတောင်းခြင်း
            evolved_code = self.ask_ai_to_fix(issue_log, original_code, target_file)

            if evolved_code and evolved_code != original_code:
                with open(target_file, 'w') as f:
                    f.write(evolved_code)
                print(f"🚀 [EVOLVED]: {target_file} has been updated. Verifying...")
                time.sleep(2) # System အခြေကျအောင် ခဏစောင့်မယ်
            else:
                print(f"❌ [STALLED]: AI could not provide a better version.")
                break

if __name__ == "__main__":
    import time
    
    if not GEMINI_API_KEY:
        print("❌ [CRITICAL]: GEMINI_API_KEY not found in environment.")
        sys.exit(1)

    engine = EvolutionEngine(GEMINI_API_KEY)
    
    # ပြင်ချင်တဲ့ ဖိုင်စာရင်း
    targets = ["app.py", "main.py", "sync_data.py"]
    
    for target in targets:
        engine.run_and_evolve(target)
