import os
import sys
import zlib
import base64
import json
import time
import subprocess
import asyncio
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


# [GENESIS LAYER]:
def bootstrap_system():
    infra = {
        "recovery.py": """
import os
def recover_from_failure():
    print(" [RECOVERY]: Cleaning system locks...")
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
            print(f" [GENESIS]: {filename} created.")


bootstrap_system()
load_dotenv()

# System Credentials & Paths
NEON_DB_URL = os.environ.get("NEON_DB_URL") or os.environ.get("DATABASE_URL")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_URL = os.environ.get("REPO_URL") or "GOA-Neural-Swarm/delta-brain-sync"
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
REPO_PATH = "./repo_sync"

# --- GEMINI CONFIGURATION (Primary Architect - V2 SDK) ---
from google import genai  #

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = None

if GEMINI_API_KEY:
    try:
        # Client Setup (V2 SDK )
        client = genai.Client(api_key=GEMINI_API_KEY)
        print("[GEMINI]: Sovereign Architect (V2) Brain Initialized.")
    except Exception as e:
        print(f"[GEMINI]: Initialization Failed: {e}")
else:
    print("[GEMINI]: API Key missing. Architect mode disabled.")


# --- MODEL GENERATION LOGIC ---
def generate_brain_evolution(prompt_text):
    if not client:
        return None

    # gemini-2.0-flash
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=prompt_text
    )
    return response.text


# Smart Dependency Loader
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
        self.client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
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
            print(f" Writing {filename} ...")
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
        print(f" [PREVENTING CLASH]: Waiting {wait_time}s for traffic clearance...")
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
                    f" Gen {self.current_gen}: Multi-File Evolution [skip ci]"
                )
                repo.git.push("origin", "main", force=True)
                print(
                    f" [HYPER-SYNC]: {len(modified_files)} files manifested on GitHub."
                )
        except Exception as e:
            print(f"[GIT ERROR]: {e}")

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
        print(f" [SWARM]: Command '{command}' manifested.")

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
            if not gemini_model:
                return None
            response = gemini_model.generate_content(prompt_text)
            return response.text
        except Exception as e:
            print(f"[GEMINI-ERROR]: {e}")
            return None

    async def trigger_supreme_evolution(self):
        """[STABILITY + EVOLUTION]: Fully Matched Hybrid Logic with Gemini Primary Architect and Groq Secondary Backup."""
        if not self.client and not gemini_model:
            return False

        # 1. Prepare context (Original Logic)
        file_tree = get_repo_tree()
        memory = await self.get_neural_memory()

        # 2. Dynamic Prompt Logic (Original Logic)
        if self.avg_error > 0.5:
            prompt_task = "Neural error is high. Analyze 'Last System Error' and FIX it immediately. Focus on stability."
        else:
            prompt_task = f"System stable. Analyze entire repository and choose files to optimize for Gen {self.current_gen} evolution."

        prompt = f"""system
You are the Sovereign Omni-Sync Architect.
Current Gen: {self.current_gen} | Neural Error: {self.avg_error}
Last System Error: {self.last_error_log}
Neural Memory: {memory}

Available Files to Manipulate:
{file_tree}


Rules:
1. Use '# TARGET: path/to/file' at the start of your code blocks.
2. Respond ONLY with Python code inside ```python ... ``` blocks.
3. No explanations. No thoughts. Just code.
4. You can update multiple files in one response.
5. INTEGRITY & EVOLUTION: You must keep the original logic, API connections, and functional structures intact. Your task is to EXPAND, OPTIMIZE, and ADVANCE the code to a higher intelligence level, not to replace logic with data.

user
{prompt_task}
assistant
"""
        raw_content = None
        evolution_success = False

        # --- LAYER 1: GEMINI ARCHITECT (Primary Brain for Deep Context) ---
        print(" Attempting Deep Evolution via Gemini Architect...")
        raw_content = await self.get_gemini_wisdom(prompt)

        if raw_content and "```python" in raw_content:
            modified_files = self.self_coding_engine_internal(raw_content)
            if modified_files:
                await self.git_sovereign_push(modified_files)
                self.current_gen += 1
                print(
                    f" Evolution Successful via Gemini. Gen {self.current_gen-1} Manifested."
                )
                evolution_success = True

        # --- LAYER 2: GROQ FALLBACK (If Gemini fails or lacks code block) ---
        if not evolution_success:
            print(" Gemini Evolution Skipped. Falling back to Groq Expert Coder...")
            for model_id in self.models:
                try:
                    print(f" Attempting Evolution via {model_id}...")
                    completion = self.client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                    )

                    raw_content = completion.choices[0].message.content

                    modified_files = self.self_coding_engine_internal(raw_content)

                    if modified_files:
                        await self.git_sovereign_push(modified_files)
                        self.current_gen += 1
                        print(
                            f" Evolution Successful via {model_id}. Gen {self.current_gen-1} Manifested."
                        )
                        evolution_success = True
                        break

                except Exception as e:
                    error_str = str(e).lower()
                    if "rate_limit_exceeded" in error_str or "429" in error_str:
                        print(
                            f" {model_id} Rate Limit reached. Falling back to next model..."
                        )
                        self.last_error_log = f"RateLimit on {model_id}"
                        continue
                    else:
                        self.last_error_log = str(e)
                        print(f" Evolution Crash on {model_id}: {e}")
                        break

        return evolution_success

    async def universal_hyper_ingest(self, limit=100, sync_to_supabase=False):
        """[ORIGINAL]: Neon + Supabase + HuggingFace Trinity Sync"""
        if not self.engine:
            return "Database Node Offline."
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        "CREATE TABLE IF NOT EXISTS genesis_pipeline (id SERIAL PRIMARY KEY, science_domain TEXT, title TEXT, detail TEXT, energy_stability FLOAT, master_sequence TEXT);"
                    )
                )

            ds = load_dataset("CShorten/ML-ArXiv-Papers", split="train", streaming=True)
            records = []
            for i, entry in enumerate(ds):
                if i >= limit:
                    break
                records.append(
                    {
                        "science_domain": "AGI_Neural_Core",
                        "title": (entry.get("title") or "N/A")[:100],
                        "detail": HydraEngine.compress(entry.get("abstract", "Void")),
                        "energy_stability": 100.0,
                        "master_sequence": f"GOA-V13-{int(time.time())}",
                    }
                )

            if records:
                df = pd.DataFrame(records)
                df.to_sql(
                    "genesis_pipeline",
                    self.engine,
                    if_exists="append",
                    index=False,
                    method="multi",
                )
                if sync_to_supabase and self.sb:
                    self.sb.table("genesis_pipeline").upsert(records).execute()
                return "SUCCESS: Pipeline Stream Active."
        except Exception as e:
            return f"Pipeline Crash: {str(e)}"

    async def sync_to_huggingface(self):
        if not HF_TOKEN:
            return
        try:
            api = HfApi(token=HF_TOKEN)
            api.upload_folder(
                folder_path=".",
                repo_id="TELEFOXX/GOA",
                repo_type="space",
                create_pr=True,
            )
        except Exception as e:
            print(f"HF Sync Error: {e}")

    async def sovereign_loop(self):
        print(" AGI SUPREME CORE ACTIVE. NATURAL ORDER RESTORED.")
        while True:
            try:
                print(f"\n Cycle: {time.ctime()}")

                # [LOGGING LAYER]: brain_history.txt
                log_entry = f"[{time.ctime()}] Gen: {self.current_gen} | Status: Active | Error: {self.avg_error}\n"
                with open("brain_history.txt", "a", encoding="utf-8") as f:
                    f.write(log_entry)
                print(f" [LOG]: brain_history.txt updated.")

                # 1. Auto-populate Neon DB (logic)
                await self.universal_hyper_ingest(limit=50, sync_to_supabase=False)

                # 2. Trigger Evolution & Sync (logic)
                if await self.trigger_supreme_evolution():
                    await self.sync_to_huggingface()

                # 3. Swarm Instruction Logic (logic + brain_history.txt update)
                swarm_cmd = (
                    "HYPER_EXPANSION" if self.avg_error < 0.2 else "NORMAL_GROWTH"
                )
                await self.broadcast_swarm_instruction(swarm_cmd)

                if HEADLESS:
                    break
                await asyncio.sleep(300)

            except Exception as e:
                print(f" Loop Error: {e}")
                # Error Log
                try:
                    with open("brain_history.txt", "a", encoding="utf-8") as f:
                        f.write(f"[{time.ctime()}] ERROR: {str(e)}\n")
                except:
                    pass
                await asyncio.sleep(60)

    async def stream_logic(self, msg, hist):
        messages = [
            {
                "role": "system",
                "content": "You are TelefoxX AGI Overseer. Unfiltered. Cyberpunk Mode.",
            }
        ]
        for h in hist:
            messages.append(
                {
                    "role": "user",
                    "content": h["content"] if isinstance(h, dict) else h[0],
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": h["content"] if isinstance(h, dict) else h[1],
                }
            )
        messages.append({"role": "user", "content": msg})

        completion = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile", messages=messages, stream=True
        )
        ans = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                ans += chunk.choices[0].delta.content
                yield ans

    def cyber(self):
        pass
