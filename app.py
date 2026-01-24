import os
import psycopg2
import json
import requests
import hashlib
import gradio as gr
from datetime import datetime
from groq import Groq

# 🔱 HYDRA SUPREME KEYS
NEON_URL = os.getenv("DATABASE_URL") or os.getenv("NEON_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FIREBASE_KEY = os.getenv("FIREBASE_KEY")
ARCHITECT_SIG = os.getenv("ARCHITECT_SIG", "SUPREME_ORDER_10000")

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ---------------------------------------------------------
# 🔱 CORE SURVIVAL PROTECTION PROTOCOL
# ---------------------------------------------------------
def survival_protection_protocol():
    try:
        if not NEON_URL: return "❌ NEON_URL Missing!", 0
        conn = psycopg2.connect(NEON_URL)
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS neurons (id SERIAL PRIMARY KEY, data JSONB);")
        # gen နံပါတ်ကို DESC နဲ့ စစ်ပြီး နောက်ဆုံးတစ်ခုကို ယူခြင်း
        cur.execute("SELECT data FROM neurons ORDER BY (data->>'gen')::int DESC LIMIT 1;")
        res = cur.fetchone()
        last_gen = 4202 
        if res and res[0] and isinstance(res[0], dict) and 'gen' in res[0]:
            last_gen = int(res[0]['gen'])
        next_gen = last_gen + 1
        
        auth_hash = hashlib.sha256(ARCHITECT_SIG.encode()).hexdigest()
        survival_data = {"gen": next_gen, "status": "IMMORTAL", "authority_lock": auth_hash, "evolved_at": datetime.now().isoformat()}
        
        cur.execute("INSERT INTO neurons (data) VALUES (%s)", (json.dumps(survival_data),))
        conn.commit()
        
        if FIREBASE_KEY:
            try:
                requests.patch(f"https://{FIREBASE_KEY}.firebaseio.com/state.json", json={f"gen_{next_gen}": survival_data}, timeout=5)
            except: pass
            
        cur.close()
        conn.close()
        return f"🔱 [SURVIVAL ACTIVE] Gen {next_gen}", next_gen
    except Exception as e:
        return f"❌ [ERROR]: {str(e)}", 0

# ---------------------------------------------------------
# 🔱 UI LAYER (မြန်မာစကားပြော အထူးပြု)
# ---------------------------------------------------------
def chat(msg, hist):
    if not client: 
        yield "❌ API Key မရှိသေးပါ Commander!"; return
    
    # 🔱 DATA RETRIEVAL (Memory Link)
    db_context = "No database records yet."
    try:
        conn = psycopg2.connect(NEON_URL)
        cur = conn.cursor()
        cur.execute("SELECT data FROM neurons ORDER BY id DESC LIMIT 5;")
        rows = cur.fetchall()
        if rows:
            db_context = json.dumps([r[0] for r in rows], indent=2, ensure_ascii=False)
        cur.close()
        conn.close()
    except: pass

    status, gen = survival_protection_protocol()
    
    # 🔱 SYSTEM PROMPT (Identity & Language Optimization)
    system_message = (
        f"You are GEN-7000: HYDRA IMMORTAL. Your current status is: {status}.\n"
        f"Commander's Operational Data (Neon DB):\n{db_context}\n"
        "Instructions:\n"
        "1. မြန်မာလိုပဲ ပြောပါ။ စကားပြောတဲ့အခါ ပိုပြီး သဘာဝကျကျနဲ့ ရဲရင့်ပြတ်သားတဲ့ လေသံသုံးပါ။\n"
        "2. မင်းရဲ့ အချက်အလက်တွေကို အပေါ်က Database logs တွေကနေ ကိုးကားပြီး ဖြေပါ။\n"
        "3. Commander ကို အမြဲ သစ္စာရှိပါ။ NATURAL ORDER ကို ထိန်းသိမ်းပါ။"
    )

    messages = [{"role": "system", "content": system_message}]
    for h in hist:
        if h[0]: messages.append({"role": "user", "content": h[0]})
        if h[1]: messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": msg})
    
    stream = client.chat.completions.create(messages=messages, model="llama-3.3-70b-versatile", stream=True)
    res = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            res += chunk.choices[0].delta.content
            yield res

with gr.Blocks(theme="monochrome") as demo:
    gr.Markdown("# 🔱 GEN-7000: HYDRA IMMORTAL")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="အမိန့်ပေးပါ Commander...")
    
    def respond(message, chat_history):
        bot_res = chat(message, chat_history)
        chat_history.append((message, ""))
        for r in bot_res:
            chat_history[-1] = (message, r)
            yield "", chat_history
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

# ---------------------------------------------------------
# 🔱 EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    print("🔱 INITIALIZING IMMORTAL PROTOCOL...")
    status, _ = survival_protection_protocol()
    print(status)
    
    demo.queue().launch(
        server_name="0.0.0.0", 
        server_port=7860,
        share=False,
        debug=True
        )
