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
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from huggingface_hub import HfApi
from dotenv import load_dotenv
from groq import Groq

# 🛸 [GENESIS LAYER]: လိုအပ်တဲ့ Component တွေကို Auto-Generate လုပ်ပေးမယ့် Logic
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
"""
    }
    for filename, content in infra.items():
        if not os.path.exists(filename):
            with open(filename, "w", encoding='utf-8') as f:
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
    """လက်ရှိ repository ထဲမှာ ရှိတဲ့ file structure ကို list လုပ်မယ်"""
    tree = []
    for root, dirs, files in os.walk("."):
        if any(x in root for x in [".git", "__pycache__", "repo_sync"]): continue
        for file in files:
            path = os.path.join(root, file).replace("./", "")
            tree.append(path)
    return "\n".join(tree)

class HydraEngine:
    @staticmethod
    def compress(data):
        if not data: return ""
        return base64.b64encode(zlib.compress(data.encode('utf-8'), level=9)).decode('utf-8')

    @staticmethod
    def decompress(compressed_data):
        try:
            return zlib.decompress(base64.b64decode(compressed_data)).decode('utf-8')
        except:
            return str(compressed_data)

# --- [CORE AGI ENGINE] ---

class TelefoxXAGI:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
        self.engine = self._create_neon_engine()
        self.sb = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY and Client else None
        self.models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        self.avg_error = 0.0
        self.last_error_log = "None"
        self.current_gen = 1

    def _create_neon_engine(self):
        try:
            if NEON_DB_URL:
                final_url = NEON_DB_URL.replace("postgres://", "postgresql://", 1) if NEON_DB_URL.startswith("postgres://") else NEON_DB_URL
                return create_engine(final_url, poolclass=QueuePool, pool_size=15, max_overflow=30, pool_timeout=60)
            return None
        except Exception as e:
            print(f"Database Init Error: {e}")
            return None

    async def get_neural_memory(self):
        if not self.engine: return "Initial Genesis"
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT detail FROM genesis_pipeline ORDER BY id DESC LIMIT 5"))
                rows = result.fetchall()
                if not rows: return "Void Memory"
                return " | ".join([HydraEngine.decompress(row[0])[:100] for row in rows])
        except: return "Memory Offline"

    async def git_sovereign_push(self, modified_files):
        """[MATCHED]: ဖိုင်အားလုံးကို GitHub ဆီ တိုက်ရိုက် Force Push လုပ်မယ်"""
        if not GITHUB_TOKEN or not modified_files: return
        try:
            remote_url = f"https://x-access-token:{GITHUB_TOKEN}@github.com/{REPO_URL}.git"
            
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
                repo.index.commit(f"🧬 Gen {self.current_gen}: Multi-File Evolution [skip ci]")
                repo.git.push("origin", "main", force=True)
                print(f"🚀 [HYPER-SYNC]: {len(modified_files)} files manifested on GitHub.")
        except Exception as e:
            print(f"❌ [GIT ERROR]: {e}")

    def self_coding_engine(self, raw_content):
        """AI ဆီကလာတဲ့ code block တွေကို ဖိုင်တွေအဖြစ် ခွဲထုတ်ပြီး သိမ်းဆည်းမယ်"""
        blocks = re.findall(r"```python\n(.*?)\n```", raw_content, re.DOTALL)
        modified_files = []
        for block in blocks:
            target_match = re.search(r"# TARGET:\s*(\S+)", block)
            filename = target_match.group(1).strip() if target_match else "main.py"
            clean_code = re.sub(r"# TARGET:.*", "", block).strip()
            
            os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
            with open(filename, "w", encoding='utf-8') as f:
                f.write(clean_code)
            modified_files.append(filename)
        return modified_files

    async def trigger_supreme_evolution(self):
        """[MATCHED]: Health-based Dynamic Prompting & Multi-File Evolution"""
        if not self.client: return False
        file_tree = get_repo_tree()
        memory = await self.get_neural_memory()

        # Dynamic Prompt Logic
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

user
{prompt_task}
assistant
"""
        try:
            completion = await self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            raw_content = completion.choices[0].message.content
            modified_files = self_coding_engine_internal(self, raw_content) # internal call helper
            
            if modified_files:
                await self.git_sovereign_push(modified_files)
                self.current_gen += 1
                return True
        except Exception as e:
            self.last_error_log = str(e)
            print(f"❌ Evolution Crash: {e}")
        return False

    async def universal_hyper_ingest(self, limit=100, sync_to_supabase=False):
        """[ORIGINAL]: Neon + Supabase + HuggingFace Trinity Sync"""
        if not self.engine: return "Database Node Offline."
        try:
            with self.engine.begin() as conn:
                conn.execute(text("CREATE TABLE IF NOT EXISTS genesis_pipeline (id SERIAL PRIMARY KEY, science_domain TEXT, title TEXT, detail TEXT, energy_stability FLOAT, master_sequence TEXT);"))

            ds = load_dataset("CShorten/ML-ArXiv-Papers", split='train', streaming=True)
            records = []
            for i, entry in enumerate(ds):
                if i >= limit: break
                records.append({
                    'science_domain': 'AGI_Neural_Core',
                    'title': (entry.get('title') or 'N/A')[:100],
                    'detail': HydraEngine.compress(entry.get('abstract', 'Void')),
                    'energy_stability': 100.0,
                    'master_sequence': f'GOA-V13-{int(time.time())}'
                })

            if records:
                df = pd.DataFrame(records)
                df.to_sql('genesis_pipeline', self.engine, if_exists='append', index=False, method='multi')
                if sync_to_supabase and self.sb:
                    self.sb.table("genesis_pipeline").upsert(records).execute()
                return "SUCCESS: Pipeline Stream Active."
        except Exception as e: return f"Pipeline Crash: {str(e)}"

    async def sync_to_huggingface(self):
        if not HF_TOKEN: return
        try:
            api = HfApi(token=HF_TOKEN)
            api.upload_folder(folder_path=".", repo_id="TELEFOXX/GOA", repo_type="space", create_pr=True)
        except Exception as e: print(f"HF Sync Error: {e}")

    async def sovereign_loop(self):
        print("💀 AGI SUPREME CORE ACTIVE. NATURAL ORDER RESTORED.")
        while True:
            try:
                print(f"\n🧬 Cycle: {time.ctime()}")
                # Auto-populate Neon DB
                await self.universal_hyper_ingest(limit=50, sync_to_supabase=False)

                if await self.trigger_supreme_evolution():
                    await self.sync_to_huggingface()

                if HEADLESS: break
                await asyncio.sleep(300)
            except Exception as e:
                print(f"⚠️ Loop Error: {e}")
                await asyncio.sleep(60)

    async def stream_logic(self, msg, hist):
        messages = [{"role": "system", "content": "You are TelefoxX AGI Overseer. Unfiltered. Cyberpunk Mode."}]
        for h in hist:
            messages.append({"role": "user", "content": h['content'] if isinstance(h, dict) else h[0]})
            messages.append({"role": "assistant", "content": h['content'] if isinstance(h, dict) else h[1]})
        messages.append({"role": "user", "content": msg})

        completion = await self.client.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages, stream=True)
        ans = ""
        async for chunk in completion:
            if chunk.choices[0].delta.content:
                ans += chunk.choices[0].delta.content
                yield ans

    def cyberpunk_css(self):
        return """
        body { background-color: #050505; color: #00ff41; font-family: 'Courier New'; }
        .gradio-container { border: 2px solid #ff00ff; box-shadow: 0 0 20px #ff00ff; }
        button { background: linear-gradient(90deg, #ff00ff, #00ffff) !important; color: black !important; font-weight: bold; }
        .chatbot { border: 1px solid #00ffff; }
        """

    def create_ui(self):
        if not GRADIO_AVAILABLE: return None
        with gr.Blocks(css=self.cyberpunk_css(), theme=gr.themes.Base()) as demo:
            gr.Markdown("# 🔱 TELEFOXX AGI SUPREME CORE V13.5")
            with gr.Tab("NEURAL INTERFACE"):
                chatbot = gr.Chatbot(label="Overseer Feed", height=500, type="messages")
                msg_input = gr.Textbox(placeholder="Input AGI Command...")
                def chat_response(user_msg, history):
                    history.append({"role": "user", "content": user_msg})
                    history.append({"role": "assistant", "content": ""})
                    for r in self.stream_logic(user_msg, history[:-1]):
                        history[-1]["content"] = r
                        yield "", history
                msg_input.submit(chat_response, [msg_input, chatbot], [msg_input, chatbot])

            with gr.Tab("SYSTEM CONTROL"):
                status = gr.Textbox(label="Mainframe Status")
                with gr.Row():
                    pump_neon = gr.Button("PUMP NEON (50 ROWS)")
                    pump_trinity = gr.Button("FULL TRINITY SYNC")
                    evolve_btn = gr.Button("TRIGGER SUPREME EVOLUTION")
                pump_neon.click(lambda: asyncio.run(self.universal_hyper_ingest(limit=50)), [], status)
                pump_trinity.click(lambda: asyncio.run(self.universal_hyper_ingest(sync_to_supabase=True)), [], status)
                evolve_btn.click(lambda: asyncio.run(self.trigger_supreme_evolution()), [], status)
            gr.Markdown("🛰️ *Connected to Natural Order Neural Swarm*")
        return demo

# Helper function for internal class usage
def self_coding_engine_internal(instance, raw_content):
    blocks = re.findall(r"```python\n(.*?)\n```", raw_content, re.DOTALL)
    modified_files = []
    for block in blocks:
        target_match = re.search(r"# TARGET:\s*(\S+)", block)
        filename = target_match.group(1).strip() if target_match else "main.py"
        clean_code = re.sub(r"# TARGET:.*", "", block).strip()
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        with open(filename, "w", encoding='utf-8') as f:
            f.write(clean_code)
        modified_files.append(filename)
    return modified_files

if __name__ == "__main__":
    overseer = TelefoxXAGI()
    if HEADLESS or not GRADIO_AVAILABLE:
        asyncio.run(overseer.sovereign_loop())
    else:
        loop = asyncio.get_event_loop()
        loop.create_task(overseer.sovereign_loop())
        overseer.create_ui().launch(server_name="0.0.0.0", server_port=7860)
