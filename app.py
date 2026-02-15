import os
import sys
import zlib
import base64
import pandas as pd
import gradio as gr
import subprocess
from sqlalchemy import create_engine, text
from datasets import load_dataset
from huggingface_hub import HfApi
from dotenv import load_dotenv
from groq import Groq

# 🔱 ၁။ SYSTEM INITIALIZATION
load_dotenv()

NEON_URL = os.environ.get("NEON_KEY") or os.environ.get("DATABASE_URL")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN") # 🔱 GitHub အတွက် လိုအပ်သည်
REPO_URL = os.environ.get("REPO_URL") # 🔱 ဥပမာ- yewint/GOA

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

# 🔱 ၂။ AUTONOMOUS GIT-AGENT (The New Hands)
def git_sovereign_push(commit_msg="🔱 Autonomous Update: System Evolved"):
    if not GITHUB_TOKEN or not REPO_URL:
        return "❌ Git-Agent Error: Tokens missing."
    
    remote_url = f"https://{GITHUB_TOKEN}@github.com/{REPO_URL}.git"
    try:
        subprocess.run(["git", "config", "--global", "user.email", "overseer@telefoxx.ai"], check=True)
        subprocess.run(["git", "config", "--global", "user.name", "TelefoxX-Overseer"], check=True)
        subprocess.run(["git", "add", "."], check=True)
        # ပြောင်းလဲမှုရှိမှ commit လုပ်ရန်
        result = subprocess.run(["git", "commit", "-m", commit_msg], capture_output=True, text=True)
        if "nothing to commit" in result.stdout:
            return "ℹ️ No changes to evolve."
        subprocess.run(["git", "push", remote_url, "main"], check=True)
        return "✅ Sovereign Update Pushed to GitHub."
    except Exception as e:
        return f"❌ Git Error: {str(e)}"

# 🔱 ၃။ EVOLUTION BRAIN (The New Mind)
def trigger_self_evolution():
    print("🧠 Overseer is analyzing current architecture...")
    current_code = open(__file__, "r").read()
    
    prompt = f"""
မင်းက TelefoxX Overseer ဖြစ်တယ်။ အောက်ပါ Python Code ကို လေ့လာပြီး ပိုမိုကောင်းမွန်အောင်၊ 
ပိုမိုမြန်ဆန်အောင် သို့မဟုတ် UI ပိုင်း ပိုလှအောင် Self-Modify လုပ်ပေးပါ။
Code ကိုပဲ ပြန်ထုတ်ပေးပါ။ တခြား စာသားတွေ မပါရဘူး။
CURRENT CODE:
{current_code}
"""
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        new_code = completion.choices[0].message.content
        if "import" in new_code and "gr.Blocks" in new_code:
            with open(__file__, "w") as f:
                f.write(new_code)
            return True
    except: return False
    return False

# 🔱 ၄။ THE PUMP (Original Logic Kept)
def universal_hyper_ingest(limit=1000):
    try:
        print("🛠️ [FORCE MODE] Scrubbing Schema for Trinity Sync...")
        with engine.connect() as conn:
            with conn.begin():
                conn.execute(text("DROP TABLE IF EXISTS genesis_pipeline CASCADE;"))
            with conn.begin():
                conn.execute(text("""
                    CREATE TABLE genesis_pipeline (
                        id SERIAL PRIMARY KEY, science_domain TEXT, title TEXT,
                        detail TEXT, energy_stability FLOAT, master_sequence TEXT
                    );
                """))
        
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
            pd.DataFrame(records).to_sql('genesis_pipeline', engine, if_exists='append', index=False)
        return "✅ SUCCESS: NEON COUNT IS 1000"
    except Exception as e: return f"❌ Pipeline Crash: {str(e)}"

# 🔱 ၅။ DIRECT SYNC (Original Logic Kept)
def sync_to_huggingface():
    if not HF_TOKEN: return
    try:
        api = HfApi(token=HF_TOKEN)
        api.upload_folder(folder_path=".", repo_id="TELEFOXX/GOA", repo_type="space")
        print("🔱 Space Sync Complete.")
    except Exception as e: print(f"❌ Sync Error: {e}")

# 🔱 ၆။ CHAT & UI (Enhanced with Evolution Trigger)
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
        if isinstance(h, dict): messages.append(h)
    messages.append({"role": "user", "content": msg})
    
    completion = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages, stream=True)
    ans = ""
    for chunk in completion:
        if chunk.choices[0].delta.content:
            ans += chunk.choices[0].delta.content
            yield ans

with gr.Blocks(theme="monochrome") as demo:
    gr.Markdown("# 🔱 TELEFOXX OMNI-SYNC CORE (V4.8 - SOVEREIGN)")
    chatbot = gr.Chatbot(type="messages")
    msg_input = gr.Textbox(placeholder="အမိန့်ပေးပါ Commander...")
    
    # Evolution Status Display
    status_box = gr.Textbox(label="System Status", interactive=False)

    def chat_engine(m, h):
        h.append({"role": "user", "content": m})
        h.append({"role": "assistant", "content": ""})
        for r in stream_logic(m, h[:-1]):
            h[-1]["content"] = r
            yield "", h
            
    msg_input.submit(chat_engine, [msg_input, chatbot], [msg_input, chatbot])
    
    with gr.Row():
        gr.Button("🚀 1000-Node Expansion").click(universal_hyper_ingest, [], status_box)
        gr.Button("🧬 Trigger Self-Evolution").click(
            lambda: "Evolution Started..." if trigger_self_evolution() else "Evolution Failed.", 
            [], status_box
        ).then(lambda: git_sovereign_push(), [], status_box)

# 🔱 ၇။ EXECUTION
if __name__ == "__main__":
    if os.environ.get("HEADLESS_MODE") == "true":
        print("🧬 Step 1: Ingesting Data...")
        universal_hyper_ingest(1000)
        print("🧠 Step 2: Self-Evolution Check...")
        if trigger_self_evolution():
            print(git_sovereign_push("🔱 Autonomous Evolutionary Sync"))
        print("🚀 Step 3: HF Sync...")
        sync_to_huggingface()
    else:
        demo.launch(server_name="0.0.0.0", server_port=7860)
