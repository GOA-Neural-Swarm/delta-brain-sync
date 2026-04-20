import os
import subprocess
import sys
import time
import json
import traceback
import requests
import git
import re
import random
import base64
import torch
from datetime import datetime

try:
    from datetime import UTC
except ImportError:
    import datetime as dt

    UTC = dt.timezone.utc

from functools import lru_cache
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# Sovereign Requirements Setup
def install_requirements():
    """Installs necessary libraries for the Sovereign Engine."""
    libs = [
        "psycopg2-binary",
        "firebase-admin",
        "bitsandbytes",
        "requests",
        "accelerate",
        "GitPython",
        "sympy==1.12",
        "numpy",
        "scikit-learn",
        "transformers",
        "torch",
    ]
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *libs, "--quiet", "--no-cache-dir"]
        )
        print("✅ [SYSTEM]: Phase 7.1 Sovereign Core & Stability Patch Ready.")
    except Exception as e:
        print(f"⚠️ Install Warning: {e}")


if __name__ == "__main__":
    if not os.environ.get("REQUIREMENTS_INSTALLED"):
        install_requirements()
        os.environ["REQUIREMENTS_INSTALLED"] = "1"

# GitHub Configuration
REPO_OWNER = "GOA-Neural-Swarm"
REPO_NAME = "delta-brain-sync"
REPO_URL = f"github.com/{REPO_OWNER}/{REPO_NAME}"
REPO_PATH = "/kaggle/working/sovereign_repo_sync"


class Brain:
    """Represents a neural brain with RNA QT45 Absorption capabilities."""

    def __init__(self):
        self.memory = np.random.rand(1000)
        self.connections = {}
        self.memory_vault = {}
        self.qt45_growth_factor = 1.618
        self.sovereign_mode = True
        self.scaler = StandardScaler()
        self.svm = SVC(kernel="rbf", C=1.0, probability=True)
        self.is_trained = False

    def learn(self, input_data, output_data):
        error = np.mean((output_data - self.memory) ** 2)
        self.memory += error * (input_data - self.memory)
        for i in range(len(self.memory)):
            if self.memory[i] > 0.5:
                self.connections[i] = "SOVEREIGN_NODE"
        return error

    def learn_ml(self, stabilities, labels):
        try:
            X = np.array(stabilities).reshape(-1, 1)
            y = np.array(labels)
            X_scaled = self.scaler.fit_transform(X)
            self.svm.fit(X_scaled, y)
            self.is_trained = True
            print("🧠 [ML]: SVM Pattern Recognition Model Synchronized.")
        except Exception as e:
            print(f"⚠️ [ML ERROR]: {e}")

    def execute_natural_absorption(self, category, sequence, stability):
        data_id = len(self.memory_vault)
        stab_val = stability if stability is not None else 0.0
        seq_val = sequence if sequence is not None else "ACTG"
        cat_val = category if category is not None else "UNKNOWN"

        self.memory_vault[data_id] = {
            "cat": cat_val,
            "seq": seq_val,
            "stab": stab_val,
        }
        factor = abs(stab_val) / 500.0
        self.memory *= self.qt45_growth_factor + factor
        self.memory = np.clip(self.memory, 0.0, 1.0)
        print(f"🔱 [NATURAL ORDER]: TARGET {cat_val} ABSORBED.")


# Initialize
brain = Brain()

if not os.path.exists("main.py"):
    with open("main.py", "w") as f:
        f.write("# Initial Sovereign Main\nimport os\n")

# Load Model
print("⏳ Loading LLM Pipeline...")
try:
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        pipe = pipeline(
            "text-generation",
            model="unsloth/llama-3-8b-instruct-bnb-4bit",
            model_kwargs={"quantization_config": bnb_config},
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        print("⚠️ No GPU detected. Falling back to CPU-compatible model.")
        pipe = pipeline(
            "text-generation",
            model="HuggingFaceM4/tiny-random-LlamaForCausalLM",
            device_map=None,
            device=-1,
        )
except Exception as e:
    print(f"⚠️ Pipeline Load Failed: {e}. Falling back to dummy logic.")
    pipe = None

current_gen = 95
while True:
    try:
        print(f"⚙️ [NEURAL BRAIN]: Training Cycle Gen {current_gen}...")

        total_error = 0
        for i in range(10):
            input_sample, target_sample = np.random.rand(1000), np.random.rand(1000)
            err = brain.learn(input_sample, target_sample)
            total_error += err
        avg_error = total_error / 10

        batch_data = [("EVO", "ACTG" * 10, random.uniform(0, 100)) for _ in range(5)]
        for category, sequence, stability in batch_data:
            brain.execute_natural_absorption(category, sequence, stability)

        main_code = ""
        if os.path.exists("main.py"):
            with open("main.py", "r") as f:
                main_code = f.read()

        needs_security_patch = any(x in main_code for x in ["os.system", "os.execv"])
        target_file = "main.py" if needs_security_patch else "brain.py"
        system_task = (
            "Fix vulnerabilities" if needs_security_patch else "Optimize Brain class"
        )

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are Sovereign AI Overseer. 
Rule 1: Use ONLY '# TARGET: {target_file}' at the start of your code block.
Rule 2: Respond ONLY with Python code inside markdown python blocks (python ... ).
Rule 3: No explanations.
Current Gen: {current_gen} | Error: {avg_error}
{system_task}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

        if pipe:
            result = pipe(prompt, max_new_tokens=1000, do_sample=True, temperature=0.7)
            full_text = result[0]["generated_text"]
            assistant_part = full_text.split(
                "<|start_header_id|>assistant<|end_header_id|>"
            )[-1]

            # Fixed Regex to capture content inside triple backticks
            code_match = re.search(r"python\s*(.*?)\s*", assistant_part, re.DOTALL)

            final_code = None
            if code_match:
                final_code = code_match.group(1).strip()
            elif f"# TARGET: {target_file}" in assistant_part:
                # Fallback if AI forgets backticks but includes the target header
                final_code = assistant_part.strip()

            if final_code:
                with open(target_file, "w") as f:
                    f.write(final_code)
                print(f"💾 [FILESYSTEM]: {target_file} updated by AI.")
            else:
                print("⚠️ [AI]: No valid code block generated.")

        current_gen += 1
        time.sleep(30)

    except Exception as e:
        print(f"🚨 [CORE CRASH]: {traceback.format_exc()}")
        time.sleep(10)
        continue
