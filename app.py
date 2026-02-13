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

# 🔱 ၁။ SYSTEM INITIALIZATION (Workflow & Environment Matched)
load_dotenv()

# Workflow ထဲက NEON_KEY ကို ဦးစားပေးယူပြီး URL အဖြစ် သတ်မှတ်သည်
NEON_URL = os.environ.get("NEON_KEY") or "postgresql://neondb_owner:npg_QUqg12MzNxnI@ep-divine-river-ahpf8fzb-pooler.c-3.us-east-1.aws.neon.tech/neondb?sslmode=require"
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
                try:
                    conn.execute(text("DROP TABLE IF EXISTS genesis_pipeline CASCADE;"))
                    conn.execute(text("DROP VIEW IF EXISTS genesis_pipeline CASCADE;"))
                    print("✅ Core status cleared.")
                except Exception as e:
                    print(f"Bypassing cleanup error: {e}")

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
            
            with engine.connect() as conn:
                count = conn.execute(text("SELECT count(*) FROM genesis_pipeline")).scalar()
                return f"✅ SUCCESS: NEON COUNT IS {count} (Expansion Ready for Sync)"
        return "⚠️ Fetch Fail."
    except Exception as e:
        return f"❌ Pipeline Crash: {str(e)}"

# 🔱 ၃။ DIRECT SYNC (Security Validated for WRITE access)
def sync_to_huggingface():
    # Environment မှ Token ကို တိုက်ရိုက်စစ်ဆေးသည်
    if not HF_TOKEN: 
        print("❌ No HF_TOKEN found in Environment Secrets.")
        return
    try:
        api = HfApi()
        print("🔱 Triggering Force Sync to Space Core...")
        api.upload_folder(
            folder_path=".",
            repo_id="TELEFOXX/GOA",
            repo_type="space",
            token=HF_TOKEN,
            commit_message="🔱 GOA TRINITY-SYNC: NEURAL EVOLUTION [EXPANDED]",
            revision="main",
            create_pr=False,
            ignore_patterns=[".git*", "__pycache__*"]
        )
        print("🔱 Space Sync Complete.")
    except Exception as e:
        print(f"❌ HF Sync Forbidden: {e}")
        print("💡 Tip: Ensure HF_TOKEN is in Repository Secrets (not just Org) and has WRITE role.")

# 🔱 ၄။ OMNI-OVERSEER CHAT LOGIC (DESC Order Matched)
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
    chatbot = gr.Chatbot()
    msg_input = gr.Textbox(placeholder="အမိန့်ပေးပါ Commander...")
    
    def user(m, h): return "", h + [[m, None]]
    def bot(h):
        for r in stream_logic(h[-1][0], h[:-1]):
            h[-1][1] = r
            yield h
            
    msg_input.submit(user, [msg_input, chatbot], [msg_input, chatbot], queue=False).then(bot, chatbot, chatbot)
    gr.Button("🚀 Trigger 1000-Node Expansion").click(lambda: universal_hyper_ingest(1000), [], gr.Textbox())

# 🔱 ၆။ EXECUTION (Workflow Step 1)
if __name__ == "__main__":
    # GitHub Actions တွင် HEADLESS_MODE ကို သုံး၍ အလိုအလျောက် Run စေသည်
    if os.environ.get("HEADLESS_MODE") == "true":
        print("🧬 Trinity Step 1: Ingesting Data...")
        print(universal_hyper_ingest(limit=1000))
        print("🚀 Trinity Step 2: Syncing to Space...")
        sync_to_huggingface()
        sys.exit(0)
    else:
        # Local သို့မဟုတ် Space တွင် UI Launch သည်
        demo.launch(server_name="0.0.0.0", server_port=7860)
