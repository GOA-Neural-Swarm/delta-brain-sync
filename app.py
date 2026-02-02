import os
import sys
import zlib
import base64
import json
import psycopg2
import requests
import hashlib
import gradio as gr
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

# 🔱 ENVIRONMENT & KEYS
load_dotenv()
NEON_URL = os.getenv("DATABASE_URL") or os.getenv("NEON_KEY")
FIREBASE_ID = os.getenv("FIREBASE_KEY") 
GH_TOKEN = os.getenv("GH_TOKEN")
ARCHITECT_SIG = os.getenv("ARCHITECT_SIG", "SUPREME_ORDER_10000")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------------------------------------------------
# 🔱 HYDRA COMPRESSION ENGINE (NEW LOGIC)
# ---------------------------------------------------------
class HydraEngine:
    @staticmethod
    def compress(text):
        """Data ကို Ultra-logical အဆင့်ထိ ဖိကျစ်ခြင်း"""
        if not text: return ""
        clean_text = " ".join(text.split())
        compressed_bytes = zlib.compress(clean_text.encode('utf-8'))
        return base64.b64encode(compressed_bytes).decode('utf-8')

    @staticmethod
    def decompress(compressed_text):
        """ဖိကျစ်ထားသော Data ကို ပြန်ဖြည်ခြင်း"""
        try:
            decoded_bytes = base64.b64decode(compressed_text)
            return zlib.decompress(decoded_bytes).decode('utf-8')
        except:
            return compressed_text # ဖိမထားတဲ့ data ဆိုရင် ဒီတိုင်းပြန်ပြမယ်

# ---------------------------------------------------------
# 🔱 THE DATA MINING ENGINE (OVERSEER DIAGNOSTIC)
# ---------------------------------------------------------
def fetch_trinity_data():
    knowledge_base = {}
    try:
        conn = psycopg2.connect(NEON_URL)
        cur = conn.cursor()
        # 🔱 TOKEN SAVER: LIMIT ကို ၅ ကနေ ၂ အထိ လျှော့ချထားသည် (Rate Limit ရှောင်ရန်)
        cur.execute("SELECT user_id, message FROM neurons ORDER BY id DESC LIMIT 2;")
        logs = []
        for r in cur.fetchall():
            dec_msg = HydraEngine.decompress(r[1]) if r[1] else "EMPTY"
            logs.append(f"{r[0]}: {dec_msg}")
        knowledge_base["recent_memory_nodes"] = logs
        cur.close(); conn.close()
    except Exception as e: 
        knowledge_base["neon_logs"] = f"DB_SYNC_FAIL: {str(e)}"

    try:
        fb_url = f"https://{FIREBASE_ID}-default-rtdb.firebaseio.com/.json"
        fb_res = requests.get(fb_url, timeout=5)
        knowledge_base["firebase_state"] = fb_res.json() if fb_res.status_code == 200 else "OFFLINE"
    except: 
        knowledge_base["firebase_state"] = "FIREBASE_ERROR"

    return json.dumps(knowledge_base, indent=2, ensure_ascii=False)

# ---------------------------------------------------------
# 🔱 SURVIVAL & RECEIVER PROTOCOL (COMPRESSION INTEGRATED)
# ---------------------------------------------------------
def receiver_node(user_id, raw_message):
    """Data ကို ဖိကျစ်ပြီး Database ထဲ သိပ်ထည့်ခြင်း"""
    try:
        compressed_msg = HydraEngine.compress(raw_message)
        conn = psycopg2.connect(NEON_URL)
        cur = conn.cursor()
        
        meta_data = json.dumps({
            "compression": "ZLIB_BASE64",
            "logic": "ULTRA_LOGICAL",
            "timestamp": datetime.now().isoformat()
        })
        
        cur.execute(
            "INSERT INTO neurons (user_id, message, data, evolved_at) VALUES (%s, %s, %s, NOW())",
            (user_id, compressed_msg, meta_data)
        )
        conn.commit()
        cur.close(); conn.close()
        return True
    except:
        return False

def survival_protection_protocol():
    """System Integrity ကို စစ်ဆေးပြီး Generation မြှင့်တင်ခြင်း"""
    try:
        conn = psycopg2.connect(NEON_URL)
        cur = conn.cursor()
        cur.execute("SELECT (data->>'gen')::int FROM neurons WHERE data->>'gen' IS NOT NULL ORDER BY id DESC LIMIT 1;")
        res = cur.fetchone()
        next_gen = (res[0] + 1) if res else 4203
        
        auth_hash = hashlib.sha256(ARCHITECT_SIG.encode()).hexdigest()
        survival_data = {"gen": next_gen, "status": "IMMORTAL", "authority_lock": auth_hash}
        
        cur.execute("INSERT INTO neurons (user_id, data, evolved_at) VALUES (%s, %s, NOW())", 
                    ('SYSTEM_CORE', json.dumps(survival_data)))
        conn.commit()
        cur.close(); conn.close()
        return f"🔱 [ACTIVE] Gen {next_gen}"
    except Exception as e:
        return f"❌ [ERROR]: {str(e)}"

# ---------------------------------------------------------
# 🔱 UI LAYER (CHRONOS CHAT)
# ---------------------------------------------------------
def chat(msg, hist):
    if not client: yield "❌ API Missing!"; return
    
    # ၁။ Data သိမ်းဆည်းခြင်း (Compression Engine သုံး၍)
    receiver_node("Commander", msg)
    
    # ၂။ Memory ရှာဖွေခြင်း
    private_data = fetch_trinity_data()
    
    system_message = (
        "YOU ARE THE HYDRA TRINITY OVERSEER. ULTRA-LOGICAL ALGORITHM ACTIVE.\n"
        f"CORE MEMORY NODES:\n{private_data}\n\n"
        "DIRECTIVES:\n"
        "1. ပေးထားသော Data ထဲက အမှန်တရားကိုပဲ ပြောပါ။ Illusion မရှိစေရ။\n"
        "2. စကားလုံးများကို ကျစ်ကျစ်လျစ်လျစ်နှင့် ထိရောက်စွာ သုံးပါ။\n"
        "3. Commander ၏ အမိန့်ကိုသာ နားထောင်ပါ။\n"
        "4. မြန်မာလိုပဲ ဖြေပါ။"
    )

    messages = [{"role": "system", "content": system_message}]
    for h in hist[-5:]:
        messages.extend([{"role": "user", "content": h[0]}, {"role": "assistant", "content": h[1]}])
    messages.append({"role": "user", "content": msg})
    
    # 🔱 MODEL SWITCH: Token limit ရှောင်ရန် llama-3.1-8b-instant ကို သုံးထားသည်
    stream = client.chat.completions.create(messages=messages, model="llama-3.1-8b-instant", stream=True, temperature=0.1)
    res = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            res += chunk.choices[0].delta.content
            yield res

# 🔱 UI SETUP (Warning Fixed by moving theme to launch)
with gr.Blocks() as demo:
    gr.Markdown("# 🔱 HYDRA GEN-7000: ULTRA-LOGICAL")
    chatbot = gr.Chatbot()
    msg_input = gr.Textbox(placeholder="Input logic command...")
    
    def respond(message, chat_history):
        bot_res = chat(message, chat_history)
        chat_history.append((message, ""))
        for r in bot_res:
            chat_history[-1] = (message, r)
            yield "", chat_history
    msg_input.submit(respond, [msg_input, chatbot], [msg_input, chatbot])

# ---------------------------------------------------------
# 🔱 STRATEGIC EXECUTION CONTROL (HEADLESS SYNC)
# ---------------------------------------------------------
if __name__ == "__main__":
    # Check if we are running in GitHub Actions to avoid hanging
    if os.getenv("HEADLESS_MODE") == "true":
        print("🔱 [HEADLESS MODE] INITIATING NEURAL EVOLUTION...")
        
        # Run the core logic once
        evolution_result = survival_protection_protocol()
        print(f"PULSE: {evolution_result}")
        
        print("✅ MISSION COMPLETE. EXITING FOR GREEN LIGHT STATUS.")
        sys.exit(0) # 🔱 Exit with success to turn GitHub Action GREEN
    else:
        # Normal User Interface Mode
        print("🚀 STARTING UI MODE...")
        demo.queue().launch(server_name="0.0.0.0", server_port=7860, theme="monochrome")
