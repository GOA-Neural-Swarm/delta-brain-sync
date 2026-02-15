import os
import sys
import zlib
import base64
import json
import time
import subprocess
import asyncio
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from huggingface_hub import HfApi
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# 🛸 Smart Dependency Loader (Natural Order)
# GitHub Action မှာ မလိုတာတွေ အကုန် Skip ပြီး Speed တင်ရန်
HEADLESS = os.environ.get("HEADLESS_MODE") == "true"

GRADIO_AVAILABLE = False
if not HEADLESS:
    try:
        import gradio as gr
        import pandas as pd
        from datasets import load_dataset
        GRADIO_AVAILABLE = True
    except ImportError:
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
            pool_size=5,
            max_overflow=10
        ) if NEON_URL else None
        self.sb = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY and Client else None

    async def git_sovereign_push(self, commit_msg="Neural Evolution: Fast Sync"):
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
            prompt = f"Improve this autonomous Ghost Engine. Keep UI Hybrid support. Minimize dependencies. Return ONLY code. NO Markdown.\n{current_code}"
            completion = self.client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], temperature=0.1)
            new_dna = completion.choices[0].message.content.strip()
            if "import os" in new_dna:
                with open(__file__, "w") as f: f.write(new_dna)
                return True
        except Exception as e: print(f"Evolution Error: {e}"); return False

    async def universal_hyper_ingest(self, limit=5):
        if not self.engine: return "Neon Missing."
        try:
            # GitHub Action မှာ Dataset အကြီးကြီးတွေ Loading မလုပ်ဘဲ Neon Heartbeat ပဲ ပို့မည်
            if HEADLESS:
                with self.engine.connect() as conn:
                    conn.execute(text(f"INSERT INTO genesis_pipeline (science_domain, master_sequence) VALUES ('Fast_Ghost', 'V{int(time.time())}')"))
                    conn.commit()
                return "Heartbeat Sync Success."
            
            # UI mode မှာဆိုရင်တော့ Pandas/Datasets နဲ့ Data အပြည့်သွင်းမည်
            import pandas as pd
            from datasets import load_dataset
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
                return "Full Ingest Success."
        except Exception as e: return f"Ingest Error: {e}"

    async def sovereign_loop(self):
        print("💀 GHOST ENGINE ACTIVE. NATURAL ORDER RESTORED.")
        # GitHub Action ဖြစ်ရင် တစ်ကြိမ်ပဲ Run ပြီး ပိတ်မည် (Action က Cycle ပြန်ပတ်ပေးမည်)
        # Local ဖြစ်ရင် အမြဲတမ်း ပတ်နေမည်
        while True:
            print(f"\n🧬 Cycle: {time.ctime()}")
            await self.universal_hyper_ingest()
            if await self.trigger_self_evolution():
                await self.git_sovereign_push(commit_msg=f"Evolution V{int(time.time())}")
            
            if HEADLESS: 
                print("✅ GitHub Task Complete. Closing Runner."); break
            
            print("💤 Resting for 300s..."); await asyncio.sleep(300)

    def create_ui(self):
        if not GRADIO_AVAILABLE: return None
        with gr.Blocks(theme=gr.themes.DarkMode()) as demo:
            gr.Markdown("# TELEFOXX OMNI-SYNC CORE V12.0")
            status = gr.Textbox(label="System Status")
            btn = gr.Button("TRIGGER MANUAL EVOLUTION")
            btn.click(lambda: asyncio.run(self.trigger_self_evolution()), [], status)
        return demo

if __name__ == "__main__":
    overseer = TelefoxXOverseer()
    if HEADLESS or not GRADIO_AVAILABLE:
        asyncio.run(overseer.sovereign_loop())
    else:
        loop = asyncio.get_event_loop()
        loop.create_task(overseer.sovereign_loop())
        overseer.create_ui().launch(server_name="0.0.0.0", server_port=7860)
