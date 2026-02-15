import os
import sys
import zlib
import base64
import json
import time
import subprocess
import asyncio
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from huggingface_hub import HfApi
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# 🛸 Smart Dependency Loader (Natural Order)
HEADLESS = os.environ.get("HEADLESS_MODE") == "true"
GRADIO_AVAILABLE = False

try:
    import gradio as gr
    from datasets import load_dataset
    GRADIO_AVAILABLE = True
except ImportError:
    if not HEADLESS:
        print("⚠️ Optional UI Libraries missing. Ghost Engine active.")

try:
    from supabase import create_client, Client
except ImportError:
    Client = None

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
        self.engine = create_engine(
            NEON_URL,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20
        ) if NEON_URL else None
        self.sb = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY and Client else None
        self.models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]

    async def git_sovereign_push(self, commit_msg="Neural Evolution: Integrity Sync"):
        if not GITHUB_TOKEN or not REPO_URL: return "Error: Credentials missing."
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
        try:
            with open(__file__, "r") as f: current_code = f.read()
            prompt = f"You are TelefoxX Overseer. Improve this Python code. UI must be high-end Cyberpunk. Return ONLY code. NO Markdown. NO ```python tags.\nCODE:\n{current_code}"
            
            new_dna = None
            for model_id in self.models:
                try:
                    print(f"🧠 Reasoning via {model_id}...")
                    completion = self.client.chat.completions.create(model=model_id, messages=[{"role": "user", "content": prompt}], temperature=0.1)
                    clean_code = completion.choices[0].message.content.strip()
                    if "```" in clean_code:
                        clean_code = clean_code.split("```")[1]
                        if clean_code.startswith("python"): clean_code = clean_code[6:]
                    clean_code = clean_code.strip()
                    if "import os" in clean_code and "gr.Blocks" in clean_code:
                        new_dna = clean_code
                        break
                except: continue

            if new_dna:
                with open(__file__, "w") as f: f.write(new_dna)
                return True
        except Exception as e: print(f"Evolution Error: {e}")
        return False

    async def universal_hyper_ingest(self, limit=50, sync_to_supabase=False):
        if not self.engine: return "Neon Missing."
        try:
            print(f"🔱 Universal Ingest (Supabase: {sync_to_supabase})...")
            # For Headless, we do a quick Neon heartbeat
            if HEADLESS:
                with self.engine.connect() as conn:
                    conn.execute(text(f"INSERT INTO genesis_pipeline (science_domain, master_sequence) VALUES ('Ghost_Evolution', 'V{int(time.time())}')"))
                    conn.commit()
            
            # For UI or Full Sync
            ds = load_dataset("CShorten/ML-ArXiv-Papers", split='train', streaming=True)
            records = []
            for i, entry in enumerate(ds):
                if i >= limit: break
                records.append({
                    'science_domain': 'Neural_Evolution',
                    'title': entry.get('title', 'N/A')[:100],
                    'detail': HydraEngine.compress(entry.get('abstract', '')),
                    'energy_stability': 100.0,
                    'master_sequence': f'GOA-V{int(time.time())}'
                })
            
            if records:
                pd.DataFrame(records).to_sql('genesis_pipeline', self.engine, if_exists='append', index=False)
                if sync_to_supabase and self.sb:
                    self.sb.table("genesis_pipeline").upsert(records).execute()
                return "SUCCESS: PIPELINE ACTIVE"
        except Exception as e: return f"Pipeline Crash: {str(e)}"

    async def sync_to_huggingface(self):
        if not HF_TOKEN: return
        try:
            api = HfApi(token=HF_TOKEN)
            api.upload_folder(folder_path=".", repo_id="TELEFOXX/GOA", repo_type="space", create_pr=True)
        except: pass

    async def sovereign_loop(self):
        print("🔱 INITIALIZING ETERNAL EVOLUTION CYCLE...")
        while True:
            try:
                print(f"\n🛰️ --- Evolution Cycle: {time.ctime()} ---")
                await self.universal_hyper_ingest(sync_to_supabase=False)
                if await self.trigger_self_evolution():
                    await self.git_sovereign_push(commit_msg=f"Autonomous Evolution: {time.time()}")
                    await self.sync_to_huggingface()
                
                if HEADLESS: print("✅ Cycle Complete."); break
                await asyncio.sleep(300)
            except Exception as e:
                print(f"⚠️ Loop Error: {e}")
                await asyncio.sleep(60)

    def stream_logic(self, msg, hist):
        messages = [{"role": "system", "content": "You are TelefoxX Overseer. Cyberpunk Mode active."}]
        for h in hist:
            messages.append({"role": "user", "content": h['content'] if isinstance(h, dict) else h[0]})
            messages.append({"role": "assistant", "content": h['content'] if isinstance(h, dict) else h[1]})
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
        if not GRADIO_AVAILABLE: return None
        with gr.Blocks(css=self.cyberpunk_css(), theme=gr.themes.DarkMode()) as demo:
            gr.Markdown("# TELEFOXX OMNI-SYNC CORE V12.8")
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
    if HEADLESS or not GRADIO_AVAILABLE:
        asyncio.run(overseer.sovereign_loop())
    else:
        loop = asyncio.get_event_loop()
        loop.create_task(overseer.sovereign_loop())
        overseer.create_ui().launch(server_name="0.0.0.0", server_port=7860)
