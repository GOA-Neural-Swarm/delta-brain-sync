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

load_dotenv()

NEON_URL = os.environ.get("NEON_KEY") or os.environ.get("DATABASE_URL")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_URL = os.environ.get("REPO_URL") or "GOA-Neural-Swarm/delta-brain-sync"

class HydraEngine:
    @staticmethod
    def compress(data):
        if not data: return ""
        return base64.b64encode(zlib.compress(data.encode('utf-8'))).decode('utf-8')
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
            pool_timeout=30
        ) if NEON_URL else None

    async def git_sovereign_push(self, commit_msg="Neural Evolution: Integrity Sync"):
        if not GITHUB_TOKEN or not REPO_URL:
            return "Git-Agent Error: Credentials missing."
        
        remote_url = f"https://{GITHUB_TOKEN}@github.com/{REPO_URL}.git"
        try:
            subprocess.run(["git", "config", "--global", "user.email", "overseer@telefoxx.ai"], check=True)
            subprocess.run(["git", "config", "--global", "user.name", "TelefoxX-Overseer"], check=True)
            
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "stash"], check=True)
            subprocess.run(["git", "pull", remote_url, "main", "--rebase"], check=True)
            subprocess.run(["git", "stash", "pop"], check=False)
            
            subprocess.run(["git", "add", "."], check=True)
            res = subprocess.run(["git", "commit", "-m", commit_msg], capture_output=True, text=True)
            if "nothing to commit" in res.stdout:
                return "No changes."
                
            subprocess.run(["git", "push", remote_url, "main", "--force"], check=True)
            return "Sovereign Update Pushed to GitHub."
        except Exception as e:
            return f"Git Critical Error: {str(e)}"

    async def trigger_self_evolution(self):
        print("Overseer analyzing architecture...")
        if not self.client: return False
        
        models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        try:
            with open(__file__, "r") as f:
                current_code = f.read()
                
            prompt = f"You are TelefoxX Overseer. Improve this Python code. UI must be high-end Cyberpunk. Return ONLY code. NO Markdown/Burmese in code body.\nCODE:\n{current_code}"
            
            for model_id in models:
                try:
                    print(f"Attempting Evolution via {model_id}...")
                    completion = self.client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1
                    )
                    new_code = completion.choices[0].message.content
                    clean_code = new_code.replace("", "").replace("", "").strip()
                    
                    if "import os" in clean_code and "gr.Blocks" in clean_code:
                        with open(__file__, "w") as f:
                            f.write(clean_code)
                        print(f"Evolution Successful: {model_id}")
                        return True
                except Exception as e:
                    if "rate_limit" in str(e): continue
                    raise e
        except Exception as e:
            print(f"Evolution Error: {e}")
        return False

    async def universal_hyper_ingest(self, limit=1000):
        if not self.engine: return "Neon Connection Missing."
        try:
            print("Rebuilding Schema...")
            with self.engine.connect() as conn:
                with conn.begin():
                    conn.execute(text("DROP TABLE IF EXISTS genesis_pipeline CASCADE;"))
                    conn.execute(text("""
                        CREATE TABLE genesis_pipeline (
                            id SERIAL PRIMARY KEY,
                            science_domain TEXT,
                            title TEXT,
                            detail TEXT,
                            energy_stability FLOAT,
                            master_sequence TEXT
                        );
                    """))
            
            ds = load_dataset("CShorten/ML-ArXiv-Papers", split='train', streaming=True)
            records = []
            for i, entry in enumerate(ds):
                if i >= limit: break
                records.append({
                    'science_domain': 'Neural_Evolution',
                    'title': entry.get('title', 'N/A'),
                    'detail': HydraEngine.compress(entry.get('abstract', '')),
                    'energy_stability': 100.0,
                    'master_sequence': 'GOA-INTEGRITY'
                })
            
            if records:
                df = pd.DataFrame(records)
                df.to_sql('genesis_pipeline', self.engine, if_exists='append', index=False, method='multi', chunksize=500)
                return "SUCCESS: 1000 NODES ACTIVE IN NEON"
        except Exception as e:
            return f"Pipeline Crash: {str(e)}"

    async def sync_to_huggingface(self):
        if not HF_TOKEN: return
        try:
            api = HfApi(token=HF_TOKEN)
            api.upload_folder(
                folder_path=".",
                repo_id="TELEFOXX/GOA",
                repo_type="space",
                create_pr=True,
                commit_message="GOA Integrity Sync",
                ignore_patterns=[".git*", "__pycache__*", "node_modules*", "*.tmp"]
            )
            print("HF Sync Successful.")
        except Exception as e:
            print(f"Sync Error: {e}")

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
        return """
        body { background-color: #050505; color: #00ff41; font-family: 'Courier New', Courier, monospace; }
        .gradio-container { border: 2px solid #ff00ff !important; box-shadow: 0 0 20px #ff00ff; border-radius: 0px !important; }
        button { background: linear-gradient(90deg, #ff00ff, #00ffff) !important; color: black !important; font-weight: bold !important; border-radius: 0px !important; }
        footer { display: none !important; }
        """

    def create_ui(self):
        with gr.Blocks(css=self.cyberpunk_css(), theme=gr.themes.DarkMode()) as demo:
            gr.Markdown("# TELEFOXX OMNI-SYNC CORE V6.2")
            
            with gr.Tab("NEURAL INTERFACE"):
                chatbot = gr.Chatbot(label="Overseer Feed", height=500, type="messages")
                msg_input = gr.Textbox(placeholder="Input command to TelefoxX...")
                
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
                    pump_btn = gr.Button("PUMP DATA")
                    evolve_btn = gr.Button("TRIGGER EVOLUTION")
                    sync_btn = gr.Button("TRINITY SYNC")

                pump_btn.click(lambda: asyncio.run(self.universal_hyper_ingest()), [], status)
                evolve_btn.click(lambda: asyncio.run(self.trigger_self_evolution()), [], status)
                sync_btn.click(lambda: asyncio.run(self.sync_to_huggingface()), [], status)

        return demo

if __name__ == "__main__":
    overseer = TelefoxXOverseer()
    
    if os.environ.get("HEADLESS_MODE") == "true":
        async def run_all():
            print("Launching Headless Sovereign Mode...")
            await overseer.universal_hyper_ing
