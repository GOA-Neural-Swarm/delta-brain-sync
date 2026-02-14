import os
import sys
import zlib
import base64
import pandas as pd
import gradio as gr
from sqlalchemy import create_engine, text
from datasets import load_dataset
from huggingface_hub import HfApi
from dotenv import load_dotenv
from groq import Groq

# 🔱 ၁။ SYSTEM INITIALIZATION
load_dotenv()

# Secrets ချိတ်ဆက်ခြင်း
NEON_URL = os.environ.get("NEON_KEY") or os.environ.get("DATABASE_URL") or "postgresql://neondb_owner:npg_QUqg12MzNxnI@ep-divine-river-ahpf8fzb-pooler.c-3.us-east-1.aws.neon.tech/neondb?sslmode=require"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
engine = create_engine(NEON_URL)

class HydraEngine:
    @staticmethod
    def compress(data):
        if not data: return ""
        return base64.b64encode(zlib.compress(data.encode('utf-8'))).decode('utf-8')
    @staticmethod
    def decompress(c):
        try: return zlib.decompress(base64.b64decode(c)).decode('utf-8')
        except: return str(c)

# 🔱 ၂။ THE PUMP: 1000-NODE TRINITY PREP
def universal_hyper_ingest(limit=1000):
    try:
        print("🛠️ [FORCE MODE] Scrubbing Schema for Trinity Sync...")
        with engine.connect() as conn:
            with conn.begin():
                conn.execute(text("DROP TABLE IF EXISTS genesis_pipeline CASCADE;"))
                print("✅ Core status cleared.")
            with conn.begin():
                print("🏗️ Rebuilding Genesis Core Table...")
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
        
        print(f"📡 Fetching Intelligence (Target: {limit} Neurons)...")
        ds = load_dataset("CShorten/ML-ArXiv-Papers", split='train', streaming=True)
        records = []
        for i, entry in enumerate(ds):
            if i >= limit: break
            records.append({
                'science_domain': 'Global_Expansion',
                'title': entry.get('title', 'N/A'),
                'detail': HydraEngine.compress(entry.get('abstract', '')),
                'energy_stability': -500.0,
                'master_sequence': 'GOA-SYNC'
            })

        if records:
            df = pd.DataFrame(records)
            with engine.begin() as conn:
                df.to_sql('genesis_pipeline', conn, if_exists='append', index=False)
            return f"✅ SUCCESS: NEON COUNT IS 1000"
    except Exception as e:
        return f"❌ Pipeline Crash: {str(e)}"

# 🔱 ၃။ DIRECT SYNC (403 BYPASS LOGIC)
def sync_to_huggingface():
    if not HF_TOKEN: 
        print("❌ No HF_TOKEN found.")
        return
    try:
        api = HfApi(token=HF_TOKEN)
        repo_id = "TELEFOXX/GOA"
        print(f"🔱 Triggering Force Sync to {repo_id}...")
        
        # နည်းလမ်း ၁ - Direct Upload
        try:
            api.upload_folder(
                folder_path=".",
                repo_id=repo_id,
                repo_type="space",
                commit_message="🔱 GOA TRINITY-SYNC: FINAL EVOLUTION",
                ignore_patterns=[".git*", "__pycache__*"]
            )
            print("🔱 Space Sync Complete via Direct Push.")
        except Exception as e:
            # နည်းလမ်း ၂ - 403 ဖြစ်ခဲ့ရင် PR ဖွင့်ပြီး အတင်းဝင်မယ်
            print(f"⚠️ Direct Push Forbidden, attempting via Pull Request...")
            api.upload_folder(
                folder_path=".",
                repo_id=repo_id,
                repo_type="space",
                create_pr=True,
                commit_message="🔱 GOA TRINITY-SYNC: BYPASS MODE"
            )
            print("🔱 PR Created. Please merge it on HF Space.")
    except Exception as e:
        print(f"❌ Final Sync Error: {e}")

# 🔱 ၄။ CHAT LOGIC
def fetch_neon_context():
    try:
        with engine.connect() as conn:
            query = text("SELECT science_domain, detail FROM genesis_pipeline ORDER BY id DESC LIMIT 5")
            rows = conn.execute(query).fetchall()
            return " | ".join([f"[{r[0]}]: {HydraEngine.decompress(r[1])}" for r in rows])
    except: return "Standby Mode"

def stream_logic(msg, hist):
    context = fetch_neon_context()
    sys_msg = f"CONTEXT: {context}\nမင်းက TelefoxX Overseer ဖြစ်တယ်။ မြန်မာလိုပဲ ဖြေဆိုပါ။"
    messages = [{"role": "system", "content": sys_msg}]
    for h in hist:
        if h[0]: messages.append({"role": "user", "content": h[0]})
        if h[1]: messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": msg})
    
    completion = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages, stream=True)
    ans = ""
    for chunk in completion:
        if chunk.choices[0].delta.content:
            ans += chunk.choices[0].delta.content
            yield ans

# 🔱 ၅။ UI SETUP
with gr.Blocks(theme="monochrome") as demo:
    gr.Markdown("# 🔱 TELEFOXX OMNI-SYNC CORE (V2.1)")
    chatbot = gr.Chatbot(type="messages") # Gradio 6 Ready
    msg_input = gr.Textbox(placeholder="အမိန့်ပေးပါ Commander...")
    
    def user(m, h): return "", h + [{"role": "user", "content": m}]
    def bot(h):
        for r in stream_logic(h[-1]["content"], h[:-1]):
            if h[-1].get("role") != "assistant":
                h.append({"role": "assistant", "content": r})
            else:
                h[-1]["content"] = r
            yield h
            
    msg_input.submit(user, [msg_input, chatbot], [msg_input, chatbot], queue=False).then(bot, chatbot, chatbot)
    gr.Button("🚀 Trigger 1000-Node Expansion").click(lambda: universal_hyper_ingest(1000), [], gr.Textbox())

# 🔱 ၆။ EXECUTION
if __name__ == "__main__":
    if os.environ.get("HEADLESS_MODE") == "true":
        print("🧬 Trinity Step 1: Ingesting Data...")
        print(universal_hyper_ingest(limit=1000))
        print("🚀 Trinity Step 2: Syncing to Space...")
        sync_to_huggingface()
        sys.exit(0)
    else:
        demo.launch(server_name="0.0.0.0", server_port=7860)
