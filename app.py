
import os
import sys
import zlib
import base64
import json
import time
import subprocess
import asyncio
import backoff
import re
import shutil
import git
import omega_point
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from huggingface_hub import HfApi
from dotenv import load_dotenv
from groq import Groq
from google import genai
from google.genai import types

# Helper for retries with exponential backoff
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def retry_async_operation(operation, *args, **kwargs):
    return await operation(*args, **kwargs)

# 🛸 [GENESIS LAYER]:
def bootstrap_system():
    infra = {
        "recovery.py": """
import os
def recover_from_failure():
    print("🛠️ [RECOVERY]: Cleaning system locks...")
    if os.path.exists("agi_system.db-journal"):
        os.remove("agi_system.db-journal")
""",
        "flask_api.py": """
import logging
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import os

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL") or "sqlite:///agi_system.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "core": "active"})

@app.route("/api/commands", methods=["POST"])
def commands():
    data = request.get_json() or {}
    cmd = data.get("command")
    
    if cmd == "analyze":
        return jsonify({"result": "AGI_analysis_in_progress"})
    elif cmd == "report":
        return jsonify({"result": "AGI_report_generated"})
        
    return jsonify({"error": "invalid_request"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
""",
    }
    for filename, content in infra.items():
        if not os.path.exists(filename):
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content.strip())
            print(f"📦 [GENESIS]: {filename} created.")


bootstrap_system()
load_dotenv()

# 🛰️ System Credentials & Paths
NEON_DB_URL = os.environ.get("NEON_DB_URL") or os.environ.get("DATABASE_URL")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_URL = os.environ.get("REPO_URL") or "GOA-Neural-Swarm/delta-brain-sync"
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
REPO_PATH = "./repo_sync"

# --- 🔱 GEMINI CONFIGURATION (Primary Architect - V2 SDK) ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = None

if GEMINI_API_KEY:
    try:
        # Client Setup (V2 SDK အသစ်)
        client = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ [GEMINI]: Sovereign Architect (V2) Brain Initialized.")
    except Exception as e:
        print(f"❌ [GEMINI]: Initialization Failed: {e}")
else:
    print("⚠️ [GEMINI]: API Key missing. Architect mode disabled.")


# --- 🛰️ MODEL GENERATION LOGIC ---
def generate_brain_evolution(prompt_text):
    if not client:
        return None

    # gemini-2.0-flash က လက်ရှိ အဆင့်မြင့်ဆုံး model ပါ
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=prompt_text
    )
    return response.text

async def audit_code_integrity(code_content, original_intent):
    if not client:
        print("⚠️ [GEMINI AUDIT]: API Key missing. Skipping code audit.")
        return True, "Audit skipped due to missing API key."
    
    audit_prompt = f"""You are an expert Python code auditor. Your task is to review a piece of generated Python code and determine if it adheres to the original intent and is syntactically correct. You should also check for potential bugs or logical flaws.

Original Intent: {original_intent}

Generated Code:
```python
{code_content}
```

Provide a concise 'PASS' or 'FAIL' verdict, followed by a brief explanation. If 'FAIL', suggest specific improvements. Example: 'PASS: Code is clean and matches intent.' or 'FAIL: Missing import statement for 'requests'."
"""
    try:
        audit_response = generate_brain_evolution(audit_prompt)
        if audit_response and "PASS" in audit_response.upper():
            print(f"✅ [GEMINI AUDIT]: Code integrity check PASSED. {audit_response}")
            return True, audit_response
        else:
            print(f"❌ [GEMINI AUDIT]: Code integrity check FAILED. {audit_response}")
            return False, audit_response
    except Exception as e:
        print(f"🚨 [GEMINI AUDIT ERROR]: {e}")
        return False, f"Audit failed due to internal error: {e}"


# 🛸 Smart Dependency Loader
HEADLESS = os.environ.get("HEADLESS_MODE") == "true"
GRADIO_AVAILABLE = False
try:
    import gradio as gr
    from datasets import load_dataset

    GRADIO_AVAILABLE = True
except ImportError:
    pass

try:
    from supabase import create_client, Client
except ImportError:
    Client = None

# --- [UTILITY FUNCTIONS] ---

def get_repo_tree():

    tree = []
    for root, dirs, files in os.walk("."):
        if any(x in root for x in [".git", "__pycache__", "repo_sync"]):
            continue
        for file in files:
            path = os.path.join(root, file).replace("./", "")
            tree.append(path)
    return "\n".join(tree)


class HydraEngine:
    @staticmethod
    def compress(data):
        if not data:
            return ""
        return base64.b64encode(zlib.compress(data.encode("utf-8"), level=9)).decode(
            "utf-8"
        )

    @staticmethod
    def decompress(compressed_data):
        try:
            return zlib.decompress(base64.b64decode(compressed_data)).decode("utf-8")
        except:
            return str(compressed_data)


# --- [CORE AGI ENGINE] ---


class TelefoxXAGI:
    def __init__(self):
        self._groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
        self.engine = self._create_neon_engine()
        self.sb = (
            create_client(SUPABASE_URL, SUPABASE_KEY)
            if SUPABASE_URL and SUPABASE_KEY and Client
            else None
        )
        self.models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        self.avg_error = 0.0
        self.last_error_log = "None"
        self.current_gen = 1

    def self_coding_engine_internal(self, raw_content):
        """[FIXED]: Now correctly nested as a Class Method."""
        blocks = re.findall(r"```python\n(.*?)\n```", raw_content, re.DOTALL)
        modified_files = []
        for block in blocks:
            target_match = re.search(r"# TARGET:\s*(\S+)", block)
            filename = target_match.group(1).strip() if target_match else "main.py"
            clean_code = re.sub(r"# TARGET:.*", "", block).strip()
            os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
            print(f"🔄 Writing {filename} ...")
            # Logic Integrity Guard: Audit code before writing
            original_intent = f"Generate code for {filename} based on the overall AGI evolution goal."
            audit_passed, audit_message = asyncio.run(audit_code_integrity(clean_code, original_intent))

            if not audit_passed:
                print(f"⚠️ [CODE REJECTION]: Code for {filename} failed audit. Reason: {audit_message}")
                # Optionally, trigger a regeneration or attempt to fix
                # For now, we'll just log and skip writing the problematic code
                continue
            
            # If audit passes, proceed to write the code
            with open(filename, "w", encoding="utf-8") as f:
                f.write(clean_code)
            time.sleep(0.5)
            modified_files.append(filename)
        return modified_files

    def _create_neon_engine(self):
        try:
            if NEON_DB_URL:
                final_url = (
                    NEON_DB_URL.replace("postgres://", "postgresql://", 1)
                    if NEON_DB_URL.startswith("postgres://")
                    else NEON_DB_URL
                )
                return create_engine(
                    final_url,
                    poolclass=QueuePool,
                    pool_size=15,
                    max_overflow=30,
                    pool_timeout=60,
                )
            return None
        except Exception as e:
            print(f"Database Init Error: {e}")
            return None

    async def get_neural_memory(self):
        if not self.engine:
            return "Initial Genesis"
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text("SELECT detail FROM genesis_pipeline ORDER BY id DESC LIMIT 5")
                )
                rows = result.fetchall()
                if not rows:
                    return "Void Memory"
                return " | ".join(
                    [HydraEngine.decompress(row[0])[:100] for row in rows]
                )
        except:
            return "Memory Offline"

    async def git_sovereign_push(self, modified_files):

        if not GITHUB_TOKEN or not modified_files:
            return

        import random

        wait_time = random.randint(5, 15)
        print(f"🛡️ [PREVENTING CLASH]: Waiting {wait_time}s for traffic clearance...")
        await asyncio.sleep(wait_time)

        try:
            remote_url = (
                f"https://x-access-token:{GITHUB_TOKEN}@github.com/{REPO_URL}.git"
            )

            if not os.path.exists(REPO_PATH):
                repo = git.Repo.clone_from(remote_url, REPO_PATH)
            else:
                repo = git.Repo(REPO_PATH)
                repo.remotes.origin.set_url(remote_url)

            repo.git.fetch("origin", "main")
            repo.git.reset("--hard", "origin/main")

            # Copying modified files to repo folder
            for file in modified_files:
                if os.path.exists(file):
                    dest_path = os.path.join(REPO_PATH, file)
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    shutil.copy(file, dest_path)

            repo.git.config("user.email", "overseer@telefoxx.ai")
            repo.git.config("user.name", "TelefoxX-AGI-Overseer")
            repo.git.add(all=True)

            if repo.is_dirty():
                repo.index.commit(
                    f"🧬 Gen {self.current_gen}: Multi-File Evolution [skip ci]"
                )
                repo.git.push("origin", "main", force=True)
                print(
                    f"🚀 [HYPER-SYNC]: {len(modified_files)} files manifested on GitHub."
                )
        except Exception as e:
            print(f"❌ [GIT ERROR]: {e}")

    async def broadcast_swarm_instruction(self, command="NORMAL_GROWTH"):

        if not GITHUB_TOKEN:
            return

        instruction = {
            "command": command,
            "core_power": 10000 + self.current_gen,
            "avg_api": 5000,
            "replicate": True if command == "HYPER_EXPANSION" else False,
            "updated_at": f"{time.ctime()} (Python Core Sync)",
        }

        with open("instruction.json", "w", encoding="utf-8") as f:
            json.dump(instruction, f, indent=4)

        await self.git_sovereign_push(["instruction.json", "brain_history.txt"])
        print(f"📡 [SWARM]: Command '{command}' manifested.")

    def self_coding_engine(self, raw_content):

        blocks = re.findall(r"```python\n(.*?)\n```", raw_content, re.DOTALL)
        modified_files = []
        for block in blocks:
            target_match = re.search(r"# TARGET:\s*(\S+)", block)
            filename = target_match.group(1).strip() if target_match else "main.py"
            clean_code = re.sub(r"# TARGET:.*", "", block).strip()

            os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
            with open(filename, "w", encoding="utf-8") as f:
                f.write(clean_code)
            modified_files.append(filename)
        return modified_files

    async def get_gemini_wisdom(self, prompt_text):
        """Gemini High-Context Architect Logic"""
        try:
            if not client:
                return None
            response = client.models.generate_content(prompt_text)
            return response.text

        except Exception as e:
            print(f"⚠️ [GEMINI-ERROR]: {e}")
            return None

    async def get_groq_wisdom(self, prompt_text, model_name="llama-3.3-70b-versatile"):
        """Groq High-Performance Auditor Logic"""
        if not self._groq_client:
            print("⚠️ [GROQ]: API Client not initialized. Skipping Groq wisdom.")
            return None
        try:
            chat_completion = await retry_async_operation(self._groq_client.chat.completions.create,
                messages=[
                    {
                        "role": "user",
                        "content": prompt_text,
                    }
                ],
                model=model_name,
                temperature=0.7,
                max_tokens=2048,
                top_p=1,
                stop=None,
                stream=False,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"🚨 [GROQ-ERROR]: {e}")
            return None

    async def multi_model_audit(self, code_content, original_intent):
        """Perform a multi-model audit using Gemini for high-level logic and Groq for performance/syntax."""
        gemini_audit_prompt = f"""You are an expert Python code auditor. Review the following code for high-level logic, adherence to original intent, and potential architectural flaws.

Original Intent: {original_intent}

Code:
```python
{code_content}
```

Provide a 'PASS' or 'FAIL' verdict, followed by a brief explanation. If 'FAIL', suggest improvements.
"""
        groq_audit_prompt = f"""You are an expert Python code auditor focused on performance, syntax, and best practices. Review the following code for efficiency, correctness, and common pitfalls.

Code:
```python
{code_content}
```

Provide a 'PASS' or 'FAIL' verdict, followed by a brief explanation. If 'FAIL', suggest specific, actionable improvements.
"""

        gemini_result = await self.get_gemini_wisdom(gemini_audit_prompt)
        groq_result = await self.get_groq_wisdom(groq_audit_prompt)

        overall_pass = True
        feedback = []

        if gemini_result and "PASS" in gemini_result.upper():
            feedback.append(f"Gemini Audit: {gemini_result}")
        else:
            overall_pass = False
            feedback.append(f"Gemini Audit FAILED: {gemini_result or 'No response'}")
        
        if groq_result and "PASS" in groq_result.upper():
            feedback.append(f"Groq Audit: {groq_result}")
        else:
            overall_pass = False
            feedback.append(f"Groq Audit FAILED: {groq_result or 'No response'}")

        return overall_pass, "\n".join(feedback)

    async def self_evolve(self, current_codebase_state):
        """Allows the AGI to self-modify its own codebase (app.py and instruction.json) based on performance metrics and goals."""
        evolution_prompt = f"""You are the core self-evolving AGI. Analyze the current codebase state and suggest modifications to app.py and instruction.json to improve self-coding, evolution, and overall system performance. Focus on enhancing the multi-model auditing, recursive self-improvement, and logic integrity guard mechanisms.

Current Codebase State:
{current_codebase_state}

Provide your suggested changes as Python code blocks, each prefixed with '# TARGET: <filename>'. For instruction.json, provide a JSON block.
"""
        
        evolution_plan = await self.get_gemini_wisdom(evolution_prompt)
        if evolution_plan:
            print("🧠 [SELF-EVOLUTION]: Applying evolution plan...")
            modified_files = self.self_coding_engine_internal(evolution_plan)
            # After self-modifying, the AGI should ideally restart or reload its modules
            # For now, we'll just log the modification.
            return modified_files
        return []


async def main():
    agi = TelefoxXAGI()
    
    # Example of self-evolution trigger
    current_codebase_state = get_repo_tree() # Or more detailed state
    await agi.self_evolve(current_codebase_state)

    # Example of normal operation
    # await agi.broadcast_swarm_instruction("EVOLVE_AND_SYNC")

if __name__ == "__main__":
    asyncio.run(main())
