import os
import sys
import zlib
import base64
import json
import psycopg2
import requests
import hashlib
import gradio as gr
import torch
import uuid
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from PIL import Image
import io

# 🔱 [SHIELD] - OMNI-ENVIRONMENT COMPATIBILITY
HAS_VIDEO_ENGINE = False
try:
    from diffusers import StableVideoDiffusionPipeline, DiffusionPipeline, DPMSolverMultistepScheduler
    from diffusers.utils import export_to_video
    if torch.cuda.is_available():
        HAS_VIDEO_ENGINE = True
except:
    pass

load_dotenv()
NEON_URL = os.getenv("DATABASE_URL")
FIREBASE_ID = os.getenv("FIREBASE_KEY") 
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class HydraEngine:
    @staticmethod
    def compress(text):
        if not text: return ""
        compressed_bytes = zlib.compress(text.encode('utf-8'))
        return base64.b64encode(compressed_bytes).decode('utf-8')

    @staticmethod
    def decompress(compressed_text):
        try:
            return zlib.decompress(base64.b64decode(compressed_text)).decode('utf-8')
        except: 
            return str(compressed_text)

# 🔱 DATA CONTROL (STRICT RAG)
def fetch_trinity_data():
    try:
        conn = psycopg2.connect(NEON_URL)
        cur = conn.cursor()
        cur.execute("SELECT message FROM neurons WHERE user_id != 'SYSTEM_CORE' ORDER BY id DESC LIMIT 2;")
        rows = cur.fetchall()
        cur.close(); conn.close()
        
        if rows:
            context_list = [HydraEngine.decompress(r[0]) for r in rows]
            return " | ".join(context_list)
        return "No specific data found in Neon DB."
    except Exception as e: 
        return f"Database Error: {str(e)}"

def receiver_node(user_id, raw_message):
    try:
        compressed_msg = HydraEngine.compress(raw_message)
        conn = psycopg2.connect(NEON_URL)
        cur = conn.cursor()
        cur.execute("INSERT INTO neurons (user_id, message, evolved_at) VALUES (%s, %s, NOW())", (user_id, compressed_msg))
        conn.commit(); cur.close(); conn.close()
    except: pass

# 🔱 CHAT ENGINE (GROUNDED & CLEAN)
def chat(msg, hist):
    receiver_node("Commander", msg)
    context = fetch_trinity_data()
    
    system_message = (
        f"CONTEXT FROM NEON DB: {context}\n\n"
        "INSTRUCTION:\n"
        "၁။ မင်းဟာ TelefoxX Overseer ဖြစ်တယ်။\n"
        "၂။ Context ထဲမှာပါတဲ့ အချက်အလက်ကိုပဲ သုံးပြီး မြန်မာလိုဖြေပါ။\n"
        "၃။ Context ထဲမှာ မပါရင် 'ကျွန်ုပ်၏ Data matrix ထဲတွင် ဤအချက်အလက် မရှိသေးပါ' ဟု ဖြေပါ။\n"
        "၄။ စကားလုံးများကို ထပ်တလဲလဲ မပြောပါနဲ့။"
    )
    
    messages = [{"role": "system", "content": system_message}]
    for h in hist[-5:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": msg})
    
    try:
        stream = client.chat.completions.create(
            messages=messages, 
            model="llama-3.1-8b-instant", 
            temperature=0.3,
            max_tokens=600,
            stream=True
        )
        res = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                res += chunk.choices[0].delta.content
                yield res
    except Exception as e:
        yield f"⚠️ Matrix Error: {str(e)}"

def respond(message, chat_history):
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": ""})
    bot_res = chat(message, chat_history[:-1])
    for r in bot_res:
        chat_history[-1]["content"] = r
        yield "", chat_history

# 🔱 UI SETUP (RESOLVING DEPRECATION WARNINGS)
with gr.Blocks() as demo: # Removed theme from here
    gr.Markdown("# 🔱 TELEFOXX: DATA-DRIVEN MATRIX")
    with gr.Tab("Neural Chat"):
        # Explicitly setting allow_tags to avoid Gradio 6.0 warning
        chatbot = gr.Chatbot(type="messages", render_markdown=True)
        msg_input = gr.Textbox(placeholder="အမိန့်ပေးပါ Commander...")
        msg_input.submit(respond, [msg_input, chatbot], [msg_input, chatbot])

# 🔱 EXECUTION (THEME DEPLOYED HERE)
if __name__ == "__main__":
    # Moved theme to launch() to fix DeprecationWarning
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, theme="monochrome")    except Exception as e:
        yield f"⚠️ Matrix Error: {str(e)}"

def respond(message, chat_history):
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": ""})
    # bot_res သို့ နောက်ဆုံး chat_history (assistant row မပါဘဲ) ပို့သည်
    bot_res = chat(message, chat_history[:-1])
    for r in bot_res:
        chat_history[-1]["content"] = r
        yield "", chat_history

# 🔱 UI SETUP
with gr.Blocks(theme="monochrome") as demo:
    gr.Markdown("# 🔱 TELEFOXX: DATA-DRIVEN MATRIX")
    with gr.Tab("Neural Chat"):
        chatbot = gr.Chatbot(type="messages")
        msg_input = gr.Textbox(placeholder="အမိန့်ပေးပါ Commander... (Data အပေါ်မှာပဲ အခြေခံပါလိမ့်မယ်)")
        msg_input.submit(respond, [msg_input, chatbot], [msg_input, chatbot])

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
