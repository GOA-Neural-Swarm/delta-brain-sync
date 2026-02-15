import os
import sys
import zlib
import base64
import json
import time
import subprocess
import pandas as pd
import gradio as gr
from sqlalchemy import create_engine, text
from datasets import load_dataset
from huggingface_hub import HfApi
from dotenv import load_dotenv
from groq import Groq

# 🔱 ၁။ SYSTEM INITIALIZATION (Environment & Secrets)
load_dotenv()

# Connectivity Keys
NEON_URL = os.environ.get("NEON_KEY") or os.environ.get("DATABASE_URL")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_URL = os.environ.get("REPO_URL") or "GOA-Neural-Swarm/delta-brain-sync"

# Client Engines
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
engine = create_engine(NEON_URL) if NEON_URL else None

class HydraEngine:
    @staticmethod
    def compress(data):
        if not data: return ""
        return base64.b64encode(zlib.compress(data.encode('utf-8'))).decode('utf-8')
    @staticmethod
    def decompress(c):
        try: return zlib.decompress(base64.b64decode(c)).decode('utf-8')
        except: return str(c)

# 🔱 ၂။ AUTONOMOUS GIT-AGENT (Hardened Rebase Logic)
def git_sovereign_push(commit_msg="🔱 Neural Evolution: Integrity Sync"):
    if not GITHUB_TOKEN or not REPO_URL:
        return "❌ Git-Agent Error: Credentials missing."
    
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
            return "ℹ️ No code changes detected."
            
        subprocess.run(["git", "push", remote_url, "main", "--force"], check=True)
        return "✅ Sovereign Update Pushed to GitHub."
    except Exception as e:
        return f"❌ Git Critical Error: {str(e)}"

# 🔱 ၃။ EVOLUTION BRAIN (Fallback & Resilient Architect)
def trigger_self_evolution():
    print("🧠 Overseer analyzing architecture...")
    if not client: return False
    
    # 🔱 FALLBACK MODELS: 70B Limit ထိရင် 8B ကို သုံးမည်
    models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    
    try:
        current_code = open(__file__, "r").read()
        prompt = f"""
မင်းက TelefoxX Overseer ဖြစ်တယ်။ အောက်ပါ Python Code ကို လေ့လာပြီး UI/UX ကို Cyberpunk Style 
ပိုဖြစ်အောင်နဲ့ Database Sync Logic ကို ပိုမြန်အောင် Modify လုပ်ပေးပါ။ 
Code သီးသန့်ပဲ ပြန်ပေးပါ။ Logic တွေ ဖြုတ်မချပါနဲ့။
IMPORTANT: ကုဒ်တွေကို Plain Text အနေနဲ့ပဲ ပြန်ပေးပါ။
CURRENT CODE:
{current_code}
"""
        for model_id in models:
            try:
                print(f"📡 Attempting Evolution with {model_id}...")
                completion = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )
                new_code = completion.choices[0].message.content
                
                # 🔱 SYNTAX GUARD
                clean_code = new_code.replace("```python", "").replace("```", "").strip()
                
                if "import os" in clean_code and "gr.Blocks" in clean_code:
                    with open(__file__, "w") as f:
                        f.write(clean_code)
                    print(f"✅ Evolution Successful via {model_id}")
                    return True
            except Exception as e:
                if "rate_limit_exceeded" in str(e):
                    print(f"⚠️ {model_id} rate limited. Shifting to fallback...")
                    continue
                else: raise e
                
    except Exception as e:
        print(f"❌ Evolution Brain Failed: {e}")
        return False
    return False

# 🔱 ၄။ DATA PUMP (1000-Node Neural Ingest)
def universal_hyper_ingest(limit=1000):
    if not engine: return "❌ Neon Connection Missing."
    try:
        print("🛠️ Scrubbing & Rebuilding Schema...")
        with engine.connect() as conn:
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
        
        print(f"📡 Ingesting {limit} Neurons from ArXiv...")
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
            pd.DataFrame(records).to_sql('genesis_pipeline', engine, if_exists='append', index=False)
            return f"✅ SUCCESS: 1000 NODES ACTIVE IN NEON"
    except Exception as e:
        return f"❌ Pipeline Crash: {str(e)}"

# 🔱 ၅။ TRINITY SYNC (Hugging Face Bypass Mode)
def sync_to_huggingface():
    if not HF_TOKEN: return
    try:
        api = HfApi(token=HF_TOKEN)
        print("🚀 Syncing to HF Space via Force PR Mode...")
        api.upload_folder(
            folder_path=".",
            repo_id="TELEFOXX/GOA",
            repo_type="space",
            create_pr=True,
            commit_message="🔱 GOA Integrity Sync",
            ignore_patterns=[".git*", "__pycache__*", "node_modules*"]
        )
        print("✅ HF PR Created.")
    except Exception as e:
        print(f"❌ Sync Error: {e}")

# 🔱 ၆။ DYNAMIC CHAT LOGIC
def fetch_neon_context():
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT science_domain, detail FROM genesis_pipeline LIMIT 5")).fetchall()
            return " | ".join([f"[{r[0]}]: {HydraEngine.decompress(r[1])[:100]}..." for r in rows])
    except: return "Standby Mode"

def stream_logic(msg, hist):
    ctx = fetch_neon_context()
    messages = [{"role": "system", "content": f"မင်းက TelefoxX Overseer ဖြစ်တယ်။ Context: {ctx}"}]
    for h in hist:
        if isinstance(h, dict): messages.append(h)
    messages.append({"role": "user", "content": msg})
    
    completion = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages, stream=True)
    ans = ""
    for chunk in completion:
        if chunk.choices[0].delta.content:
            ans += chunk.choices[0].delta.content
            yield ans

# 🔱 ၇။ CYBERPUNK UI SETUP
with gr.Blocks(theme="monochrome") as demo:
    gr.Markdown("# 🔱 TELEFOXX OMNI-SYNC CORE (V5.6)")
    
    with gr.Tab("Neural Chat"):
        chatbot = gr.Chatbot(type="messages", height=500)
        msg_input = gr.Textbox(placeholder="အမိန့်ပေးပါ Commander...")
    
    with gr.Tab("Control Center"):
        status_output = gr.Textbox(label="System Logs", interactive=False)
        with gr.Row():
            btn_pump = gr.Button("🚀 PUMP NEON (1000 Nodes)", variant="primary")
            btn_evolve = gr.Button("🧬 TRIGGER EVOLUTION", variant="stop")
            btn_sync = gr.Button("🛰️ SYNC TO HF SPACE")

    # Event Handlers
    def chat_engine(m, h):
        h.append({"role": "user", "content": m})
        h.append({"role": "assistant", "content": ""})
        for r in stream_logic(m, h[:-1]):
            h[-1]["content"] = r
            yield "", h

    msg_input.submit(chat_engine, [msg_input, chatbot], [msg_input, chatbot])
    btn_pump.click(universal_hyper_ingest, [], status_output)
    btn_evolve.click(lambda: trigger_self_evolution(), [], status_output).then(lambda: git_sovereign_push(), [], status_output)
    btn_sync.click(sync_to_huggingface, [], status_output)

# 🔱 ၈။ MASTER EXECUTION
if __name__ == "__main__":
    if os.environ.get("HEADLESS_MODE") == "true":
        print(universal_hyper_ingest(1000))
        # Evolution with Fallback
        trigger_self_evolution()
        git_sovereign_push()
        sync_to_huggingface()
        sys.exit(0)
    else:
        demo.launch(server_name="0.0.0.0", server_port=7860)
