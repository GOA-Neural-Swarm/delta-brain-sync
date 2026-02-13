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

# 🔱 ၁။ CORE TRINITY INITIALIZATION
load_dotenv()
NEON_URL = os.getenv("DATABASE_URL") or os.getenv("NEON_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)
engine = create_engine(NEON_URL) if NEON_URL else None

class HydraEngine:
    @staticmethod
    def compress(data):
        """Data Injection မတိုင်ခင် Neural Compression လုပ်ခြင်း"""
        if not data: return ""
        return base64.b64encode(zlib.compress(data.encode('utf-8'))).decode('utf-8')

    @staticmethod
    def decompress(c):
        """Neural Data ဖတ်ရှုရန် Extraction လုပ်ခြင်း"""
        try:
            return zlib.decompress(base64.b64decode(c)).decode('utf-8')
        except:
            return str(c)

# 🔱 ၂။ NEURAL MANAGEMENT (EDIT & SYNC)
def update_neural_record(record_id, new_message):
    """Database ထဲရှိ Neural Data ကို အမှားအယွင်းမရှိ Edit လုပ်ခြင်း"""
    if not NEON_URL: return "❌ Error: Database Key Missing"
    try:
        compressed_msg = HydraEngine.compress(new_message)
        conn = psycopg2.connect(NEON_URL)
        cur = conn.cursor()
        # id ရော message ပါရှိမှ update လုပ်မည်
        cur.execute("UPDATE neurons SET message = %s WHERE id = %s", (compressed_msg, int(record_id)))
        conn.commit()
        cur.close(); conn.close()
        return f"🔱 Neural Record {record_id} has been optimized and re-synced."
    except Exception as e:
        return f"❌ System Error: {str(e)}"

def universal_hyper_ingest(domain_choice, limit=500):
    """Hugging Face မှ Science & Tech Data အားလုံးကို စုပ်ယူခြင်း"""
    try:
        # ArXiv သည် Science/Tech theories အတွက် အကောင်းဆုံးဖြစ်သည်
        ds = load_dataset("arxiv_dataset", split='train', streaming=True, trust_remote_code=True)
        records = []
        count = 0
        print(f"📡 Expanding Core via {domain_choice}...")
        
        for entry in ds:
            if count >= limit: break
            
            # ဒေတာများကို HydraEngine ဖြင့် Encode လုပ်မည်
            records.append({
                'science_domain': domain_choice,
                'title': entry.get('title'),
                'detail': HydraEngine.compress(entry.get('abstract', '')),
                'energy_stability': -500.0,
                'master_sequence': entry.get('categories')
            })
            count += 1
            
        pd.DataFrame(records).to_sql('genesis_pipeline', engine, if_exists='append', index=False)
        return f"✅ Expansion Successful: {count} {domain_choice} theories ingested."
    except Exception as e:
        return f"❌ Expansion Failed: {str(e)}"

# 🔱 ၃။ OMNI-SYNC LOGIC (THE OVERSEER)
def fetch_neon_context():
    try:
        conn = psycopg2.connect(NEON_URL, connect_timeout=5)
        cur = conn.cursor()
        # Hybrid Fetching: Neural memories နှင့် Science theories များကို ပေါင်းစပ်ယူမည်
        cur.execute("""
            (SELECT user_id, message FROM neurons ORDER BY id DESC LIMIT 5)
            UNION ALL
            (SELECT science_domain, detail FROM genesis_pipeline ORDER BY random() LIMIT 5)
        """)
        rows = cur.fetchall()
        cur.close(); conn.close()
        
        if rows:
            return " | ".join([f"[{r[0]}]: {HydraEngine.decompress(r[1])}" for r in rows])
        return "Initial Directive Active"
    except Exception as e:
        return f"Matrix Sync Standby: {str(e)}"

def stream_logic(msg, hist):
    context = fetch_neon_context()
    system_message = (
        f"MASTER CONTEXT: {context}\n\n"
        "DIRECTIVE: မင်းဟာ TelefoxX Overseer ဖြစ်တယ်။ အထက်ပါ Context ထဲက သိပ္ပံအချက်အလက်တွေကိုပဲ သုံးပါ။ "
        "မြန်မာလို ရှင်းလင်းပြတ်သားစွာ ဖြေဆိုပါ။ Hallucination လုံးဝမလုပ်ပါနဲ့။"
    )
    
    messages = [{"role": "system", "content": system_message}]
    for h in hist[-3:]: messages.append({"role": "user", "content": h['content']})
    messages.append({"role": "user", "content": msg})

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.1,
            stream=True
        )
        ans = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                ans += chunk.choices[0].delta.content
                yield ans
    except Exception as e:
        yield f"🔱 Matrix Link Lost: {str(e)}"

# 🔱 ၄။ UI SETUP (GRADIO MONOCHROME)
with gr.Blocks(theme="monochrome", title="TELEFOXX OMNI-SYNC") as demo:
    gr.Markdown(f"# 🔱 TELEFOXX OMNI-SYNC CORE\n**Intelligence Status:** Operational")
    
    with gr.Tab("Omni-Overseer"):
        chatbot = gr.Chatbot(type="messages")
        msg_input = gr.Textbox(placeholder="အမိန့်ပေးပါ Commander...", interactive=True)
        msg_input.submit(lambda m, h: (h + [{"role": "user", "content": m}], ""), [msg_input, chatbot], [chatbot, msg_input]).then(
            respond_wrapper := (lambda h: (h, h)), [chatbot], [chatbot] # Simple stream handling
        )
        # Note: Simplified respond logic for standard Gradio deployment
        def chat_interface(message, history):
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": ""})
            for r in stream_logic(message, history[:-1]):
                history[-1]["content"] = r
                yield "", history
        msg_input.submit(chat_interface, [msg_input, chatbot], [msg_input, chatbot])

    with gr.Tab("Core Configuration"):
        with gr.Row():
            target_id = gr.Number(label="Neural ID")
            update_val = gr.Textbox(label="New Intelligence String")
        update_btn = gr.Button("🔱 Execute Neural Rewrite")
        update_btn.click(update_neural_record, [target_id, update_val], gr.Textbox(label="Status"))
        
        gr.Markdown("---")
        domain_sel = gr.Dropdown(["Aerospace_Propulsion", "Quantum_Theories", "AI_Logic", "Bio_Engineering"], label="Expansion Domain")
        ingest_btn = gr.Button("🚀 Trigger Global Expansion")
        ingest_btn.click(universal_hyper_ingest, [domain_sel], gr.Textbox(label="Pipeline History"))

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
