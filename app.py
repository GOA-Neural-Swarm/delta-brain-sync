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

# 🔱 ၁။ CORE INITIALIZATION (SECURITY & NEURAL KEYS)
load_dotenv()
# မင်းပေးထားတဲ့ Neon URL ကို တိုက်ရိုက်သုံးထားတယ်
NEON_URL = "postgresql://neondb_owner:npg_QUqg12MzNxnI@ep-divine-river-ahpf8fzb-pooler.c-3.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
engine = create_engine(NEON_URL)

class HydraEngine:
    """မူလအတိုင်း Neural Data ကို Encode/Decode လုပ်သည့် အင်ဂျင်"""
    @staticmethod
    def compress(data):
        if not data: return ""
        return base64.b64encode(zlib.compress(data.encode('utf-8'))).decode('utf-8')
    @staticmethod
    def decompress(c):
        try: return zlib.decompress(base64.b64decode(c)).decode('utf-8')
        except: return str(c)

# 🔱 ၂။ DATA PIPELINE (FORCE OVERWRITE LOGIC)
def universal_hyper_ingest(limit=50):
    """Colab တွင် အောင်မြင်ခဲ့သော View-Killer Logic ဖြင့် ဒေတာသွင်းခြင်း"""
    if not engine: return "❌ Database Offline"
    try:
        print("🛠️ [FORCE MODE] Killing Existing Objects & Rebuilding Table...")
        with engine.begin() as conn:
            # Table မဟုတ်ဘဲ View ဖြစ်နေလျှင် ဖျက်ပစ်မည့် Logic
            conn.execute(text("DROP VIEW IF EXISTS genesis_pipeline CASCADE;"))
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
        
        print("📡 Fetching Intelligence from Stable Source (ML-ArXiv-Papers)...")
        # ML Papers source က ပိုမြန်ပြီး ပိုစိတ်ချရသည်
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
            print(f"📥 Buffered: {entry.get('title')[:40]}...")

        if records:
            df = pd.DataFrame(records)
            with engine.begin() as conn:
                df.to_sql('genesis_pipeline', conn, if_exists='append', index=False)
            
            with engine.connect() as conn:
                count = conn.execute(text("SELECT count(*) FROM genesis_pipeline")).scalar()
                return f"✅ SUCCESS: {count} Neural Records Synced to Neon."
        return "⚠️ Sync Failed: No Data Fetched."
    except Exception as e:
        return f"❌ Pipeline Crash: {str(e)}"

# 🔱 ၃။ DIRECT API SYNC (NO GIT ERROR)
def sync_to_huggingface():
    if not HF_TOKEN: return
    try:
        api = HfApi()
        api.upload_folder(
            folder_path=".",
            repo_id="TELEFOXX/GOA",
            repo_type="space",
            token=HF_TOKEN,
            ignore_patterns=[".git*", "__pycache__*"]
        )
        print("🔱 Space Sync Complete.")
    except Exception as e:
        print(f"❌ Sync Failed: {e}")

# 🔱 ၄။ OMNI-OVERSEER CHAT LOGIC
def fetch_neon_context():
    """Database မှ Compressed ဒေတာများကို Context အဖြစ် ပြန်ယူခြင်း"""
    try:
        with engine.connect() as conn:
            # မူလ Code ထဲက neurons table နဲ့ pipeline table နှစ်ခုလုံးကို ရှာတဲ့ logic
            query = text("""
                (SELECT science_domain as user_id, detail as message FROM genesis_pipeline ORDER BY id DESC LIMIT 3)
            """)
            rows = conn.execute(query).fetchall()
            return " | ".join([f"[{r[0]}]: {HydraEngine.decompress(r[1])}" for r in rows])
    except: return "Standby Mode"

def stream_logic(msg, hist):
    context = fetch_neon_context()
    messages = [{"role": "system", "content": f"CONTEXT: {context}\nမင်းက TelefoxX Overseer ဖြစ်တယ်။ မြန်မာလို ဖြေဆိုပါ။"}]
    for h in hist: 
        if h[0]: messages.append({"role": "user", "content": h[0]})
        if h[1]: messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": msg})
    
    # Brain ကို ပိုတော်တဲ့ Llama-3.3-70b-versatile သို့ Upgrade လုပ်ထားသည်
    completion = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages, stream=True)
    ans = ""
    for chunk in completion:
        if chunk.choices[0].delta.content:
            ans += chunk.choices[0].delta.content
            yield ans

# 🔱 ၅။ UI SETUP (GRADIO MONOCHROME)
with gr.Blocks(theme="monochrome", title="TELEFOXX OMNI-SYNC") as demo:
    gr.Markdown("# 🔱 TELEFOXX OMNI-SYNC CORE\n**Status:** Operational")
    with gr.Tab("Omni-Overseer"):
        chatbot = gr.Chatbot()
        msg_input = gr.Textbox(placeholder="အမိန့်ပေးပါ Commander...")
        def user(m, h): return "", h + [[m, None]]
        def bot(h):
            for r in stream_logic(h[-1][0], h[:-1]):
                h[-1][1] = r
                yield h
        msg_input.submit(user, [msg_input, chatbot], [msg_input, chatbot], queue=False).then(bot, chatbot, chatbot)

    with gr.Tab("Expansion Control"):
        status_box = gr.Textbox(label="Expansion Status")
        gr.Button("🚀 Trigger Global Expansion").click(universal_hyper_ingest, [], status_box)

# 🔱 ၆။ EXECUTION CONTROL
if __name__ == "__main__":
    if os.getenv("HEADLESS_MODE") == "true":
        print("🔱 TRIGGERING DATA PUMP & SYNC...")
        print(universal_hyper_ingest(limit=50))
        sync_to_huggingface()
        sys.exit(0) # Proper exit for GitHub Actions
    else:
        demo.launch(server_name="0.0.0.0", server_port=7860)
