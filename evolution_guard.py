import time
import subprocess
import os
import requests
import json
import re
import sys
import omega_point

# API Configurations
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

class Guard:
    """Main class for the guard system."""

    def __init__(self, target_script):
        """
        Initialize the guard system.

        Args:
        target_script (str): The target script to guard.
        """
        self.target_script = target_script

    def get_ai_correction(self, error_log, original_code):
        """
        Retrieve AI correction for the given error log and original code.

        Args:
        error_log (str): The error log to correct.
        original_code (str): The original code to correct.

        Returns:
        str: The corrected code.
        """
        print(" [GUARD]: AI is analyzing the error...")

        prompt = f"Fix this Python error:\n{error_log}\n\nCode:\n{original_code}\n\nReturn ONLY the clean code."

        # --- ATTEMPT 1: GEMINI (Primary) ---
        print("[GUARD-GEMINI]: Requesting correction...")
        gemini_url = f"https://api.ai21.com/studio/v1/assistants/gemini/complete"
        gemini_payload = {
            "prompt": prompt,
            "maxTokens": 2048,
            "temperature": 0.2,
        }

        headers = {
            "Authorization": f"Bearer {GEMINI_API_KEY}",
            "Content-Type": "application/json",
        }

        try:
            res = requests.post(
                gemini_url, headers=headers, json=gemini_payload, timeout=30
            )
            data = res.json()
            if res.status_code == 200 and "completions" in data:
                content = data["completions"][0]["text"]
                return re.sub(r"```python\n|```", "", content).strip()
            else:
                print(f"[GEMINI-FAIL]: Status {res.status_code}. Switching to Groq...")
        except Exception as e:
            print(f"[GEMINI-ERROR]: {e}. Switching to Groq...")

        # --- ATTEMPT 2: GROQ (Fallback) ---
        print("[GUARD-GROQ]: Requesting fallback correction...")
        groq_url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }

        groq_payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            response = requests.post(
                groq_url, headers=headers, json=groq_payload, timeout=30
            )
            data = response.json()

            if "error" in data and "rate_limit_exceeded" in str(data):
                print("[RATE-LIMIT]: Sleeping for 20 seconds...")
                time.sleep(20)
                return self.get_ai_correction(error_log, original_code)

            if "choices" in data:
                content = data["choices"][0]["message"]["content"]
                return re.sub(r"```python\n|```", "", content).strip()
            else:
                print(f"[GUARD]: API Error Response: {data}")
                return original_code

        except Exception as e:
            print(f"[GUARD]: Request failed: {e}")
            return original_code

    def run_guard(self):
        """
        Run the guard process for the given target script.
        """
        print(f"[GUARD]: Launching {self.target_script} in Observation Mode...")

        process = subprocess.Popen(
            ["python3", self.target_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        start_time = time.time()
        error_output = ""

        while time.time() - start_time < 60:
            line = process.stderr.readline()
            if line:
                print(f"[LOG]: {line.strip()}")
                error_output += line
                if "Traceback" in line or "Error" in line:
                    print("[GUARD]: Critical Error detected! Terminating and fixing...")
                    process.terminate()

                    with open(self.target_script, "r") as f:
                        original_code = f.read()

                    corrected = self.get_ai_correction(error_output, original_code)
                    with open(self.target_script, "w") as f:
                        f.write(corrected)

                    print("[GUARD]: System evolved. Restarting Guard Cycle...")
                    return self.run_guard()

            if process.poll() is not None and process.poll() != 0:
                remaining_error = process.stderr.read()
                print(f"[GUARD]: Process died with error: {remaining_error}")

                with open(self.target_script, "r") as f:
                    original_code = f.read()

                corrected_code = self.get_ai_correction(remaining_error, original_code)
                with open(self.target_script, "w") as f:
                    f.write(corrected_code)
                print("[GUARD]: Correction applied. Rebooting system...")
                return self.run_guard()

            time.sleep(1)

        print(
            "[GUARD]: System is stable and sovereign. Handing over to background process."
        )
        sys.exit(0)

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "main.py"
    guard = Guard(target)
    guard.run_guard()