import os
import sys
import zlib
import base64
import psycopg2
import pandas as pd
import gradio as gr
from sqlalchemy import create_engine, text
from datasets import load_dataset
from huggingface_hub import HfApi
from dotenv import load_dotenv
from groq import Groq

# 🔱 ၁။ SYSTEM INITIALIZATION
load_dotenv()
NEON_URL = "postgresql://neondb_owner:npg_QUqg12MzNxnI@ep-divine-river-ahpf8fzb-pooler.c-3.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

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

# 🔱 ၂။ THE PUMP: UNSTOPPABLE SCHEMA RESET (ISOLATED TRANSACTIONS)
def universal_hyper_ingest(limit=1000):
    try:
        print("🛠️ [FORCE MODE] Scrubbing Existing Schema with Savepoints...")
        with engine.connect() as conn:
            # ၁။ View ကို သီးခြား Transaction နဲ့ ဖျက်မယ်
            with conn.begin():
                try:
                    conn.execute(text("DROP VIEW IF EXISTS genesis_pipeline CASCADE;"))
                    print("✅ View status cleared.")
                except Exception as e:
                    print(f"Skipping View Drop: {e}")

            # ၂။ Table ကို သီးခြား Transaction နဲ့ ဖျက်မယ်
            with conn.begin():
                try:
                    conn.execute(text("DROP TABLE IF EXISTS genesis_pipeline CASCADE;"))
                    print("✅ Table status cleared.")
                except Exception as e:
                    print(f"Skipping Table Drop: {e}")

            # ၃။ Table အသစ်ကို လုံးဝလွတ်လပ်တဲ့ Transaction နဲ့ ဆောက်မယ်
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
        
        print("📡 Fetching Stable Intelligence (ML-ArXiv)...")
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
                return f"✅ SUCCESS: NEON COUNT IS {count}"
        return "⚠️ Fetch Fail."
    except Exception as e:
        return f"❌ Pipeline Crash: {str(e)}"

# 🔱 ၃။ DIRECT SYNC WITH FORCE OVERWRITE (NO-PR BYPASS)
def sync_to_huggingface():
    if not HF_TOKEN: return
    try:
        api = HfApi()
        print("🔱 Triggering Force Sync to Space Core...")
        # create_pr=False ကို သုံးပြီး Main Branch ကို Direct Push လုပ်ခိုင်းတာပါ
        api.upload_folder(
            folder_path=".",
            repo_id="TELEFOXX/GOA",
            repo_type="space",
            token=HF_TOKEN,
            commit_message="🔱 GOA OMNI-SYNC: NEURAL EXPANSION [FORCE PUSH]",
            revision="main",
            create_pr=False,
            ignore_patterns=[".git*", "__pycache__*"]
        )
        print("🔱 Space Sync Complete.")
    except Exception as e:
        print(f"❌ HF Sync Forbidden: {e} \n💡 Tip: Check if HF_TOKEN has 'WRITE' role and PR bypass is active.")

# 🔱 ၄။ OMNI-OVERSEER CHAT LOGIC
def fetch_neon_context():
    try:
        with engine.connect() as conn:
            query = text("SELECT science_domain, detail FROM genesis_pipeline ORDER BY id DESC LIMIT 3")
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
    gr.Markdown("# 🔱 TELEFOXX OMNI-SYNC CORE")
    chatbot = gr.Chatbot()
    msg_input = gr.Textbox(placeholder="အမိန့်ပေးပါ Commander...")
    def user(m, h): return "", h + [[m, None]]
    def bot(h):
        for r in stream_logic(h[-1][0], h[:-1]):
            h[-1][1] = r
            yield h
    msg_input.submit(user, [msg_input, chatbot], [msg_input, chatbot], queue=False).then(bot, chatbot, chatbot)
    gr.Button("🚀 Trigger Expansion").click(universal_hyper_ingest, [], gr.Textbox())

# 🔱 ၆။ EXECUTION
if __name__ == "__main__":
    if os.getenv("HEADLESS_MODE") == "true":
        print(universal_hyper_ingest(limit=50))
        sync_to_huggingface()
        sys.exit(0)
    else:
        demo.launch(server_name="0.0.0.0", server_port=7860)
