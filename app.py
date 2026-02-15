import os
import sys
import zlib
import base64
import json
import time
import subprocess
import pandas as pd
import gradio as gr
import asyncio
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from datasets import load_dataset
from huggingface_hub import HfApi
from dotenv import load_dotenv
from groq import Groq

# 🛸 Dynamic Stack Guard
try:
    from supabase import create_client, Client
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "supabase", "httpx"])
    from supabase import create_client, Client

load_dotenv()

# 🛰️ System Credentials
NEON_URL = os.environ.get("NEON_KEY") or os.environ.get("DATABASE_URL")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_URL = os.environ.get("REPO_URL") or "GOA-Neural-Swarm/delta-brain-sync"
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

class HydraEngine:
    @staticmethod
    def compress(data):
        if not data: return ""
        return base64.b64encode(zlib.compress(data.encode('utf-8'), level=9)).decode('utf-8')
    
    @staticmethod
    def decompress(c):
        try: return zlib.decompress(base64.b64decode(c)).decode('utf-8')
        except: return str(c)

class TelefoxXOverseer:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
        
        # 🔱 Neon Engine: Stability Focused
        self.engine = create_engine(
            NEON_URL,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_timeout=60,
            connect_args={'connect_timeout': 60}
        ) if NEON_URL else None
        
        # 🛰️ Supabase Client: Kept in Stack
        self.sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

    async def git_sovereign_push(self, commit_msg="Neural Evolution: Integrity Sync"):
        if not GITHUB_TOKEN or not REPO_URL:
            return "Git-Agent Error: Credentials missing."
        remote_url = f"https://{GITHUB_TOKEN}@github.com/{REPO_URL}.git"
        try:
            subprocess.run(["git", "config", "--global", "user.email", "overseer@telefoxx.ai"], check=True)
            subprocess.run(["git", "config", "--global", "user.name", "TelefoxX-Overseer"], check=True)
            subprocess.run(["git", "add", "."], check=True)
            res = subprocess.run(["git", "commit", "-m", commit_msg], capture_output=True, text=True)
            if "nothing to commit" in res.stdout: return "No changes."
            subprocess.run(["git", "push", remote_url, "main", "--force"], check=True)
            return "Sovereign Update Pushed to GitHub."
        except Exception as e: return f"Git Critical Error: {str(e)}"

    async def trigger_self_evolution(self):
        if not self.client: return False
        models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        try:
            with open(__file__, "r") as f: current_code = f.read()
            prompt = f"You are TelefoxX Overseer. Improve this Python code. UI must be high-end Cyberpunk. Return ONLY code. NO Markdown/Burmese in code body. NO ```python tags.\nCODE:\n{current_code}"
            for model_id in models:
                try:
                    completion = self.client.chat.completions.create(model=model_id, messages=[{"role": "user", "content": prompt}], temperature=0.1)
                    clean_code = completion.choices[0].message.content
                    if "```" in clean_code:
                        clean_code = clean_code.split("```")[1]
                        if clean_code.startswith("python"): clean_code = clean_code[6:]
                    clean_code = clean_code.strip()
                    if "import os" in clean_code and "gr.Blocks" in clean_code:
                        with open(__file__, "w") as f: f.write(clean_code)
                        return True
                except: continue
        except Exception as e: print(f"Evolution Error: {e}")
        return False

    async def universal_hyper_ingest(self, limit=500, sync_to_supabase=False):
        if not self.engine: return "Neon Connection Missing."
        try:
            print(f"🔱 Rebuilding Trinity Schema (Supabase Sync: {sync_to_supabase})...")
            with self.engine.connect() as conn:
                with conn.begin():
                    conn.execute(text("DROP TABLE IF EXISTS genesis_pipeline CASCADE;"))
                    conn.execute(text("CREATE TABLE genesis_pipeline (id SERIAL PRIMARY KEY, science_domain TEXT, title TEXT, detail TEXT, energy_stability FLOAT, master_sequence TEXT);"))
            
            ds = load_dataset("CShorten/ML-ArXiv-Papers", split='train', streaming=True)
            records = []
            for i, entry in enumerate(ds):
                if i >= limit: break
                records.append({
                    'science_domain': 'Neural_Evolution',
                    'title': entry.get('title', 'N/A')[:100],
                    'detail': HydraEngine.compress(entry.get('abstract', '')),
                    'energy_stability': 100.0,
                    'master_sequence': 'GOA-HYBRID-V10'
                })
            
            if records:
                # 1. Main Sync: Neon
                pd.DataFrame(records).to_sql('genesis_pipeline', self.engine, if_exists='append', index=False, method='multi', chunksize=500)
                
                # 2. Hybrid Sync: Supabase (Only if requested to prevent timeouts)
                status_msg = "SUCCESS: NEON NODES ACTIVE"
                if sync_to_supabase and self.sb:
                    print("🛰️ Syncing to Supabase...")
                    self.sb.table("genesis_pipeline").upsert(records).execute()
                    status_msg += " + SUPABASE SYNCED"
                
                return status_msg
        except Exception as e: return f"Pipeline Crash: {str(e)}"

    async def sync_to_huggingface(self):
        if not HF_TOKEN: return
        try:
            api = HfApi(token=HF_TOKEN)
            api.upload_folder(folder_path=".", repo_id="TELEFOXX/GOA", repo_type="space", create_pr=True)
            print("HF Sync Successful.")
        except Exception as e: print(f"Sync Error: {e}")

    def stream_logic(self, msg, hist):
        messages = [{"role": "system", "content": "You are TelefoxX Overseer. Cyberpunk Mode active."}]
        for h in hist:
            u = h['content'] if isinstance(h, dict) else h[0]
            a = h['content'] if isinstance(h, dict) else h[1]
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": msg})
        completion = self.client.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages, stream=True)
        ans = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                ans += chunk.choices[0].delta.content
                yield ans

    def cyberpunk_css(self):
        return "body { background-color: #050505; color: #00ff41; font-family: 'Courier New'; } .gradio-container { border: 2px solid #ff00ff; box-shadow: 0 0 20px #ff00ff; } button { background: linear-gradient(90deg, #ff00ff, #00ffff) !important; color: black !important; }"

    def create_ui(self):
        with gr.Blocks(css=self.cyberpunk_css(), theme=gr.themes.DarkMode()) as demo:
            gr.Markdown("# TELEFOXX OMNI-SYNC CORE V10.0")
            with gr.Tab("NEURAL INTERFACE"):
                chatbot = gr.Chatbot(label="Overseer Feed", height=500, type="messages")
                msg_input = gr.Textbox(placeholder="Input command...")
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
                    pump_neon = gr.Button("PUMP NEON")
                    pump_trinity = gr.Button("FULL TRINITY SYNC")
                    evolve_btn = gr.Button("TRIGGER EVOLUTION")
                
                pump_neon.click(lambda: asyncio.run(self.universal_hyper_ingest(sync_to_supabase=False)), [], status)
                pump_trinity.click(lambda: asyncio.run(self.universal_hyper_ingest(sync_to_supabase=True)), [], status)
                evolve_btn.click(lambda: asyncio.run(self.trigger_self_evolution()), [], status)
        return demo

if __name__ == "__main__":
    overseer = TelefoxXOverseer()
    if os.environ.get("HEADLESS_MODE") == "true":
        async def run_all():
            print("Launching Sovereign Mode (Hybrid)...")
            await overseer.universal_hyper_ingest(sync_to_supabase=False) # Headless mode: Neon only for safety
            if await overseer.trigger_self_evolution():
                await overseer.git_sovereign_push()
            await overseer.sync_to_huggingface()
        asyncio.run(run_all())
    else:
        overseer.create_ui().launch(server_name="0.0.0.0", server_port=7860)
