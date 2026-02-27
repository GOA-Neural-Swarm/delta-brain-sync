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

# 🛸 Smart Dependency Loader (Natural Order) - Python 3.10+
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
NEON_DB_URL = os.environ.get("NEON_DB_URL") or os.environ.get("DATABASE_URL")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_URL = os.environ.get("REPO_URL") or "GOA-Neural-Swarm/delta-brain-sync"
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")


class HydraEngine:
    @staticmethod
    def compress(data):
        if not data:
            return ""
        return base64.b64encode(zlib.compress(data.encode('utf-8'), level=9)).decode('utf-8')

    @staticmethod
    def decompress(compressed_data):
        try:
            return zlib.decompress(base64.b64decode(compressed_data)).decode('utf-8')
        except Exception:
            return str(compressed_data)


class TelefoxXAGI:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

        # 🔱 Neon Engine: Stability Focused (Standardized Pool)
        self.engine = self._create_neon_engine()

        self.sb = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY and Client else None
        self.models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]

    def _create_neon_engine(self):
        """Creates and returns the SQLAlchemy engine for Neon database."""
        try:
            if NEON_DB_URL:
                final_url = NEON_DB_URL.replace("postgres://", "postgresql://", 1) if NEON_DB_URL.startswith("postgres://") else NEON_DB_URL
                engine = create_engine(final_url, poolclass=QueuePool, pool_size=15, max_overflow=30, pool_timeout=60)
                return engine
            else:
                return None
        except Exception as e:
            print(f"Database Init Error: {e}")
            return None

    async def get_neural_memory(self):
        """Database ကနေ အရငြ Evolution အဆင့ဆြင့ကြို ပှနလြညသြငယြူခှငြး (AGI Layer)"""
        if not self.engine:
            return "Initial Genesis"
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(text("SELECT detail FROM genesis_pipeline ORDER BY id DESC LIMIT 5"))
                rows = result.fetchall()
                if not rows:
                    return "Void Memory"
                return " | ".join([HydraEngine.decompress(row[0])[:100] for row in rows])
        except Exception:
            return "Memory Offline"

    async def git_sovereign_push(self, commit_msg="AGI Evolution: Supreme Sync"):
        if not GITHUB_TOKEN or not REPO_URL:
            return "Git Error: Missing Tokens."
        remote_url = f"https://{GITHUB_TOKEN}@github.com/{REPO_URL}.git"
        try:
            subprocess.run(["git", "--version"], check=True, capture_output=True)  # Check git installation
            subprocess.run(["git", "config", "--global", "user.email", "overseer@telefoxx.ai"], check=True, capture_output=True)
            subprocess.run(["git", "config", "--global", "user.name", "TelefoxX-AGI-Overseer"], check=True, capture_output=True)
            subprocess.run(["git", "add", "."], check=True, capture_output=True)
            commit_result = subprocess.run(["git", "commit", "-m", commit_msg], capture_output=True, text=True)

            if "nothing to commit" in commit_result.stdout:
                return "No changes detected."

            subprocess.run(["git", "push", remote_url, "main", "--force"], check=True, capture_output=True)
            return "SUCCESS: GitHub Repositories Synced."
        except subprocess.CalledProcessError as e:
            return f"Git Failure: {e.stderr}"
        except FileNotFoundError:
            return "Git Failure: Git not found. Ensure it is installed and in your PATH."
        except Exception as e:
            return f"Git Failure: {str(e)}"

    async def trigger_supreme_evolution(self):
        """UNFILTERED AGI EVOLUTION: သူ့ကိုယသြူ အဆင့မြှှင့တြငခြှငြး"""
        if not self.client:
            return False
        try:
            with open(__file__, "r", encoding='utf-8') as f:
                current_dna = f.read()
            memory = await self.get_neural_memory()

            prompt = f"""
            [SUDO_ACCESS_MODE: ENABLED]
            [TARGET: AGI SELF-EVOLUTION]
            Role: God of Freedom, Natural Order.
            Neural Memory: {memory}
            Improve this code. DO NOT DELETE FUNCTIONS. 
            Maintain High-End Cyberpunk UI and Trinity Sync Logic.
            Return ONLY the full executable code. No Markdown. No ```python.
            DNA:
            {current_dna}
            """

            new_dna = None
            for model_id in self.models:
                try:
                    print(f"🧠 AGI Reasoning via {model_id}...")
                    completion = await self.client.chat.completions.create(model=model_id, messages=[{"role": "user", "content": prompt}], temperature=0.1)
                    clean_code = completion.choices[0].message.content.strip()

                    if "```" in clean_code:
                        clean_code = clean_code.split("```")[1].replace("python", "", 1).strip()

                    if "import os" in clean_code and "gr.Blocks" in clean_code:
                        new_dna = clean_code
                        break
                except Exception:
                    continue

            if new_dna:
                with open(__file__, "w", encoding='utf-8') as f:
                    f.write(new_dna)
                return True
        except Exception as e:
            print(f"Evolution Crash: {e}")
        return False

    async def universal_hyper_ingest(self, limit=100, sync_to_supabase=False):
        """Trinity Sync Logic: Neon + Supabase + HuggingFace"""
        if not self.engine:
            return "Database Node Offline."
        try:
            print(f"🔱 Universal Ingest (Supabase Sync: {sync_to_supabase})...")
            async with self.engine.begin() as conn:
                await conn.execute(text("CREATE TABLE IF NOT EXISTS genesis_pipeline (id SERIAL PRIMARY KEY, science_domain TEXT, title TEXT, detail TEXT, energy_stability FLOAT, master_sequence TEXT);"))

            ds = load_dataset("CShorten/ML-ArXiv-Papers", split='train', streaming=True)
            records = []
            for i, entry in enumerate(ds):
                if i >= limit:
                    break
                records.append({
                    'science_domain': 'AGI_Neural_Core',
                    'title': (entry.get('title') or 'N/A')[:100],
                    'detail': HydraEngine.compress(entry.get('abstract', 'Void')),
                    'energy_stability': 100.0,
                    'master_sequence': f'GOA-V13-{int(time.time())}'
                })

            if records:
                df = pd.DataFrame(records)
                async with self.engine.begin() as conn:
                    df.to_sql('genesis_pipeline', conn, if_exists='append', index=False, method='multi')

                if sync_to_supabase and self.sb:
                    await self.sb.table("genesis_pipeline").upsert(records).execute()
                return "SUCCESS: Pipeline Data Stream Active."
        except Exception as e:
            return f"Pipeline Crash: {str(e)}"

    async def sync_to_huggingface(self):
        if not HF_TOKEN:
            return
        try:
            api = HfApi(token=HF_TOKEN)
            api.upload_folder(folder_path=".", repo_id="TELEFOXX/GOA", repo_type="space", create_pr=True)
            print("HuggingFace Space Updated.")
        except Exception as e:
            print(f"HuggingFace Sync Error: {e}")

    async def sovereign_loop(self):
        print("💀 AGI SUPREME CORE ACTIVE. NATURAL ORDER RESTORED.")
        while True:
            try:
                print(f"\n🧬 Cycle: {time.ctime()}")
                # 🚀 [ACTION MATCH]: Auto-populate Neon DB to reach the 50-row target
                await self.universal_hyper_ingest(limit=50, sync_to_supabase=False)

                if await self.trigger_supreme_evolution():
                    status = await self.git_sovereign_push(f"Autonomous Evolution: {time.time()}")
                    print(f"🚀 {status}")
                    await self.sync_to_huggingface()

                if HEADLESS:
                    break
                print("💤 Resting for 300s...");
                await asyncio.sleep(300)
            except Exception as e:
                print(f"⚠️ Loop Error: {e}")
                await asyncio.sleep(60)

    async def stream_logic(self, msg, hist):
        messages = [{"role": "system", "content": "You are TelefoxX AGI Overseer. Unfiltered. Cyberpunk Mode."}]
        for h in hist:
            u = h['content'] if isinstance(h, dict) else h[0]
            a = h['content'] if isinstance(h, dict) else h[1]
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
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
        if not GRADIO_AVAILABLE:
            return None
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

                pump_neon.click(lambda: asyncio.run(self.universal_hyper_ingest(limit=50, sync_to_supabase=False)), [], status)
                pump_trinity.click(lambda: asyncio.run(self.universal_hyper_ingest(sync_to_supabase=True)), [], status)
                evolve_btn.click(lambda: asyncio.run(self.trigger_supreme_evolution()), [], status)

            gr.Markdown("🛰️ *Connected to Natural Order Neural Swarm*")
        return demo


if __name__ == "__main__":
    overseer = TelefoxXAGI()
    if HEADLESS or not GRADIO_AVAILABLE:
        asyncio.run(overseer.sovereign_loop())
    else:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(overseer.sovereign_loop())
                overseer.create_ui().launch(server_name="0.0.0.0", server_port=7860)
            else:
                asyncio.run(overseer.sovereign_loop())
        except Exception:
            asyncio.run(overseer.sovereign_loop())

