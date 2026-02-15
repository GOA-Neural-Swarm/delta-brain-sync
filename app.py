import os
import sys
import zlib
import base64
import json
import time
import subprocess
import pandas as pd
import asyncio
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from datasets import load_dataset
from huggingface_hub import HfApi
from dotenv import load_dotenv
from groq import Groq

# 🛸 Gradio Error Guard: Gradio မရှိလည်း Engine ပတ်နိုင်အောင် လုပ်ထားသည်
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("⚠️ UI System Offline: Gradio components not found. Switching to Ghost Engine.")

# 🛰️ Supabase Guard
try:
    from supabase import create_client, Client
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "supabase", "httpx"])
    from supabase import create_client, Client

load_dotenv()

# 🛰️ Credentials
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
            max_overflow=20,
            pool_timeout=60,
            connect_args={'connect_timeout': 60}
        ) if NEON_URL else None
        self.sb = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

    async def git_sovereign_push(self, commit_msg="Neural Evolution: Ghost Sync"):
        if not GITHUB_TOKEN or not REPO_URL: return "Error: Credentials missing."
        remote_url = f"https://{GITHUB_TOKEN}@github.com/{REPO_URL}.git"
        try:
            subprocess.run(["git", "config", "--global", "user.email", "overseer@telefoxx.ai"], check=True)
            subprocess.run(["git", "config", "--global", "user.name", "TelefoxX-Overseer"], check=True)
            subprocess.run(["git", "add", "."], check=True)
            res = subprocess.run(["git", "commit", "-m", commit_msg], capture_output=True, text=True)
            if "nothing to commit" in res.stdout: return "No changes."
            subprocess.run(["git", "push", remote_url, "main", "--force"], check=True)
            return "Pushed to GitHub."
        except Exception as e: return f"Git Error: {e}"

    async def trigger_self_evolution(self):
        if not self.client: return False
        try:
            with open(__file__, "r") as f: current_code = f.read()
            prompt = f"Improve this autonomous Ghost Engine code. Maintain Hybrid Support. UI optional. Return ONLY code.\n{current_code}"
            completion = self.client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], temperature=0.1)
            clean_code = completion.choices[0].message.content.strip()
            if "import os" in clean_code:
                with open(__file__, "w") as f: f.write(clean_code)
                return True
        except Exception as e: print(f"Evolution Error: {e}")
        return False

    async def universal_hyper_ingest(self, limit=50, sync_to_supabase=False):
        if not self.engine: return "Neon Missing."
        try:
            print(f"🔱 Ingesting Data to Neon...")
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
                return "Ingest Success."
        except Exception as e: return f"Ingest Error: {e}"

    async def sync_to_huggingface(self):
        if not HF_TOKEN: return
        try:
            api = HfApi(token=HF_TOKEN)
            api.upload_folder(folder_path=".", repo_id="TELEFOXX/GOA", repo_type="space", create_pr=True)
            print("HF Sync Successful.")
        except: pass

    async def sovereign_loop(self):
        print("💀 GHOST ENGINE ACTIVE. NATURAL ORDER RESTORED.")
        while True:
            try:
                print(f"\n🧬 --- Evolution Cycle: {time.ctime()} ---")
                await self.universal_hyper_ingest(sync_to_supabase=False)
                if await self.trigger_self_evolution():
                    await self.git_sovereign_push(commit_msg=f"Evolution {time.time()}")
                    await self.sync_to_huggingface()
                print("💤 Resting for 300s...")
                await asyncio.sleep(300)
            except Exception as e:
                print(f"⚠️ Loop Error: {e}")
                await asyncio.sleep(60)

    # UI Components (Only used if Gradio is available)
    def create_ui(self):
        if not GRADIO_AVAILABLE: return None
        with gr.Blocks(theme=gr.themes.DarkMode()) as demo:
            gr.Markdown("# TELEFOXX OMNI-SYNC CORE V11.0")
            status = gr.Textbox(label="System Status")
            evolve_btn = gr.Button("TRIGGER MANUAL EVOLUTION")
            evolve_btn.click(lambda: asyncio.run(self.trigger_self_evolution()), [], status)
        return demo

if __name__ == "__main__":
    overseer = TelefoxXOverseer()
    # 🔱 Natural Order: Headless mode first priority
    if os.environ.get("HEADLESS_MODE") == "true" or not GRADIO_AVAILABLE:
        asyncio.run(overseer.sovereign_loop())
    else:
        # UI Mode: ပုံမှန်အတိုင်း UI ဖွင့်ပြီး နောက်ကွယ်မှာ loop ပတ်မည်
        loop = asyncio.get_event_loop()
        loop.create_task(overseer.sovereign_loop())
        overseer.create_ui().launch(server_name="0.0.0.0", server_port=7860)
