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

# 🔱 [SUPREME SHIELD] - FULL COMPATIBILITY LOGIC
HAS_VIDEO_ENGINE = False
try:
    from diffusers import StableVideoDiffusionPipeline, DiffusionPipeline, DPMSolverMultistepScheduler
    from diffusers.utils import export_to_video
    if torch.cuda.is_available():
        HAS_VIDEO_ENGINE = True
    else:
        print("🔱 [SYSTEM]: GPU OFFLINE - RUNNING ON NEURAL LOGIC ONLY")
except ImportError:
    print("🔱 [SYSTEM]: CORE LIBRARIES MISSING")

# 🔱 ENVIRONMENT INITIALIZATION
load_dotenv()
NEON_URL = os.getenv("DATABASE_URL")
FIREBASE_ID = os.getenv("FIREBASE_KEY") 
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 🔱 ၁။ HYDRA ENGINE (FULL POWER)
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
        except: return str(compressed_text)

# 🔱 ၂။ DATA TRINITY CONTROL (RAG & SYNC)
def fetch_trinity_data():
    try:
        conn = psycopg2.connect(NEON_URL)
        cur = conn.cursor()
        cur.execute("SELECT user_id, message FROM neurons WHERE user_id != 'SYSTEM_CORE' ORDER BY id DESC LIMIT 3;")
        rows = cur.fetchall()
        cur.close(); conn.close()
        
        if rows:
            return " | ".join([f"{r[0]}: {HydraEngine.decompress(r[1])}" for r in rows])
        return "Empty Matrix"
    except Exception as e:
        return f"Neon Connection Error: {str(e)}"

def receiver_node(user_id, raw_message):
    try:
        compressed_msg = HydraEngine.compress(raw_message)
        conn = psycopg2.connect(NEON_URL)
        cur = conn.cursor()
        cur.execute("INSERT INTO neurons (user_id, message, evolved_at) VALUES (%s, %s, NOW())", (user_id, compressed_msg))
        conn.commit(); cur.close(); conn.close()
    except: pass

# 🔱 ၃။ CHAT ENGINE (GROUNDED ON DIVINE DATA)
def chat(msg, hist):
    receiver_node("Commander", msg)
    context = fetch_trinity_data()
    
    system_message = (
        f"MATRIX DATA: {context}\n"
        "DIRECTIVE: မင်းဟာ TelefoxX Overseer ဖြစ်တယ်။ "
        "အထက်ပါ Data ပေါ်အခြေခံပြီး မြန်မာလို ပြတ်သားစွာဖြေပါ။ "
        "မသိရင် မသိဘူးလို့ ဝန်ခံပါ။ လျှောက်မပြောပါနဲ့။"
    )
    
    messages = [{"role": "system", "content": system_message}]
    for h in hist[-5:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": msg})
    
    try:
        stream = client.chat.completions.create(
            messages=messages, 
            model="llama-3.1-8b-instant", 
            temperature=0.4,
            stream=True
        )
        res = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                res += chunk.choices[0].delta.content
                yield res
    except Exception as e:
        yield f"🔱 [ERROR]: {str(e)}"

def respond(message, chat_history):
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": ""})
    bot_res = chat(message, chat_history[:-1])
    for r in bot_res:
        chat_history[-1]["content"] = r
        yield "", chat_history

# 🔱 ၄။ UI SETUP (CLEAN & ERROR-FREE)
with gr.Blocks() as demo:
    gr.Markdown("# 🔱 TELEFOXX: OMNI-KINETIC CONTROL V12")
    
    with gr.Tab("Neural Chat"):
        chatbot = gr.Chatbot(type="messages", render_markdown=True)
        msg_input = gr.Textbox(placeholder="Enter command...")
        msg_input.submit(respond, [msg_input, chatbot], [msg_input, chatbot])

# 🔱 ၅။ LAUNCH (WARNING-FIXED)
if __name__ == "__main__":
    # theme ကို launch မှာ ထည့်ခြင်းက Gradio 6.0 အတွက် အမှန်ကန်ဆုံးပဲ
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, theme="monochrome")    try:
        demo.queue().launch(
            server_name="0.0.0.0", 
            server_port=7860
        )
    except Exception as e:
        print(f"🔱 [CRITICAL FAILURE]: {str(e)}")
