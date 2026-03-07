import subprocess
import os
import requests
import json
import re
import sys

# Groq API Configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def get_ai_correction(error_log, original_code):
    print("🧠 [GUARD]: AI is analyzing the error and rewriting logic...")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    
    prompt = f"""
    The following python code crashed with this error:
    --- ERROR ---
    {error_log}
    --- ORIGINAL CODE ---
    {original_code}
    
    Task: Fix the error and return ONLY the full, clean Python code. 
    No markdown, no explanations, just the code.
    """
    
    payload = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}]
    }
    
    response = requests.post(url, headers=headers, json=payload)
    content = response.json()['choices'][0]['message']['content']
    # Cleaning potential markdown
    clean_code = re.sub(r'```python\n|```', '', content).strip()
    return clean_code

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
