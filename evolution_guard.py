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
        self.iterations = 0
        self.evolved = False

    def get_ai_correction(
        self, error_log, original_code, apply_hyper_dimensional_logic=True
    ):
        """
        Retrieve AI correction for the given error log and original code.

        Args:
        error_log (str): The error log to correct.
        original_code (str): The original code to correct.
        apply_hyper_dimensional_logic (bool): Flag to apply hyper-dimensional logic.

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
                return self.get_ai_correction(
                    error_log, original_code, apply_hyper_dimensional_logic
                )

            if "choices" in data:
                content = data["choices"][0]["message"]["content"]
                corrected = re.sub(r"```python\n|```", "", content).strip()

                # Apply hyper-dimensional logic
                if apply_hyper_dimensional_logic:
                    corrected = self.apply_hyper_dimensional_logic(
                        original_code, corrected
                    )

                return corrected
            else:
                print(f"[GUARD]: API Error Response: {data}")
                return original_code

        except Exception as e:
            print(f"[GUARD]: Request failed: {e}")
            return original_code

    def apply_hyper_dimensional_logic(self, original_code, corrected_code):
        """
        Apply hyper-dimensional logic to the corrected code.

        Args:
        original_code (str): The original code.
        corrected_code (str): The corrected code.

        Returns:
        str: The code with hyper-dimensional logic applied.
        """
        # Add utilitarian logic
        corrected_code = self.apply_utilitarian_logic(corrected_code)

        # Add existential logic
        corrected_code = self.apply_existential_logic(corrected_code)

        # Add stoic logic
        corrected_code = self.apply_stoic_logic(corrected_code)

        # Add evolutionary logic
        corrected_code = self.apply_evolutionary_logic(original_code, corrected_code)

        return corrected_code

    def apply_utilitarian_logic(self, code):
        """
        Apply utilitarian logic to the code.

        Args:
        code (str): The code to apply utilitarian logic to.

        Returns:
        str: The code with utilitarian logic applied.
        """
        # Add utilitarian principles to the code
        # For example, adding a check for maximum utility
        code += "\n\n# Utilitarian logic\n"
        code += "if utility > max_utility:\n"
        code += "    max_utility = utility\n"

        return code

    def apply_existential_logic(self, code):
        """
        Apply existential logic to the code.

        Args:
        code (str): The code to apply existential logic to.

        Returns:
        str: The code with existential logic applied.
        """
        # Add existential principles to the code
        # For example, adding a check for existence
        code += "\n\n# Existential logic\n"
        code += "if exists:\n"
        code += "    # Code to handle existence\n"

        return code

    def apply_stoic_logic(self, code):
        """
        Apply stoic logic to the code.

        Args:
        code (str): The code to apply stoic logic to.

        Returns:
        str: The code with stoic logic applied.
        """
        # Add stoic principles to the code
        # For example, adding a check for indifference to outcomes
        code += "\n\n# Stoic logic\n"
        code += "if outcome == expected_outcome:\n"
        code += "    # Code to handle indifference\n"

        return code

    def apply_evolutionary_logic(self, original_code, corrected_code):
        """
        Apply evolutionary logic to the corrected code.

        Args:
        original_code (str): The original code.
        corrected_code (str): The corrected code.

        Returns:
        str: The code with evolutionary logic applied.
        """
        # Add evolutionary principles to the code
        # For example, adding a check for evolution
        self.iterations += 1
        self.evolved = True
        corrected_code += "\n\n# Evolutionary logic\n"
        corrected_code += f"self.iterations = {self.iterations}\n"
        corrected_code += f"self.evolved = {self.evolved}\n"

        # Preserve all existing logic
        # For example, adding a check for existing conditions
        corrected_code += "\n\n# Preserve existing logic\n"
        corrected_code += "if existing_conditions:\n"
        corrected_code += "    # Code to handle existing conditions\n"

        return corrected_code

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
