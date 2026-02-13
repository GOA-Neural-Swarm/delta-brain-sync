import os
import zlib
import base64
import psycopg2
import pandas as pd
import gradio as gr
from dotenv import load_dotenv
from groq import Groq
from datasets import load_dataset
from sqlalchemy import create_engine
from huggingface_hub import HfApi

# 🔱 CORE INITIALIZATION
load_dotenv()
NEON_URL = os.getenv("NEON_KEY") or os.getenv("DATABASE_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

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

# 🔱 DATABASE INGESTION
def universal_hyper_ingest(limit=50):
    if not engine: return "❌ Database Connection Missing"
    try:
        print("🚀 Fetching ArXiv Data...")
        ds = load_dataset("arxiv_dataset", "full", split='train', streaming=True)
        records = []
        for i, entry in enumerate(ds):
            if i >= limit: break
            records.append({
                'science_domain': 'Global_Expansion',
                'title': entry.get('title'),
                'detail': HydraEngine.compress(entry.get('abstract', '')),
                'energy_stability': -500.0,
                'master_sequence': entry.get('categories')
            })
        if records:
            pd.DataFrame(records).to_sql('genesis_pipeline', engine, if_exists='append', index=False)
            return f"✅ Ingested {len(records)} Records."
        return "⚠️ No records found."
    except Exception as e:
        return f"❌ Pipeline Error: {str(e)}"

# 🔱 HUGGING FACE DIRECT UPLOAD (GIT-FREE)
def sync_to_huggingface():
    if not HF_TOKEN:
        print("⚠️ HF_TOKEN Missing. Skipping Sync.")
        return
    try:
        print("📡 Direct Syncing to Hugging Face via API...")
        api = HfApi()
        api.upload_folder(
            folder_path=".",
            repo_id="TELEFOXX/GOA",
            repo_type="space",
            token=HF_TOKEN,
            ignore_patterns=[".git*", "__pycache__*"]
        )
        print("✅ Hugging Face Space Updated Successfully!")
    except Exception as e:
        print(f"❌ HF Sync Failed: {e}")

# 🔱 CHAT LOGIC & UI (မပြောင်းလဲပါ)
def fetch_neon_context():
    try:
        conn = psycopg2.connect(NEON_URL, connect_timeout=5)
        cur = conn.cursor()
        cur.execute("(SELECT user_id, message FROM neurons ORDER BY id DESC LIMIT 3) UNION ALL (SELECT science_domain, detail FROM genesis_pipeline ORDER BY id DESC LIMIT 3)")
        rows = cur.fetchall()
        cur.close(); conn.close()
        return " | ".join([f"[{r[0]}]: {HydraEngine.decompress(r[1])}" for r in rows]) if rows else "Directive Active"
    except: return "Sync Standby"

def stream_logic(msg, hist):
    context = fetch_neon_context()
    messages = [{"role": "system", "content": f"CONTEXT: {context}\nမင်းက TelefoxX Overseer ဖြစ်တယ်။ မြန်မာလိုဖြေပါ။"}]
    for h in hist: messages.extend([{"role": "user", "content": h[0]}, {"role": "assistant", "content": h[1]}])
    messages.append({"role": "user", "content": msg})
    completion = client.chat.completions.create(model="llama-3.1-8b-instant", messages=messages, stream=True)
    ans = ""
    for chunk in completion:
        if chunk.choices[0].delta.content:
            ans += chunk.choices[0].delta.content
            yield ans

with gr.Blocks() as demo:
    gr.Markdown("# 🔱 TELEFOXX OMNI-SYNC CORE")
    chatbot = gr.Chatbot()
    msg_input = gr.Textbox(placeholder="အမိန့်ပေးပါ Commander...")
    def user(m, h): return "", h + [[m, None]]
    def bot(h):
        for r in stream_logic(h[-1][0], h[:-1]):
            h[-1][1] = r
            yield h
    msg_input.submit(user, [msg_input, chatbot], [msg_input, chatbot], queue=False).then(bot, chatbot, chatbot)

if __name__ == "__main__":
    if os.getenv("HEADLESS_MODE") == "true":
        print(universal_hyper_ingest(limit=50))
        sync_to_huggingface() # ဒီမှာ တိုက်ရိုက် Sync လုပ်မယ်
        os._exit(0)
    else:
        demo.launch(server_name="0.0.0.0", server_port=7860, theme="monochrome")
