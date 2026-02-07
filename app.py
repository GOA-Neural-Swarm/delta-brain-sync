import os
import sys
import zlib
import base64
import json
import psycopg2
import requests
import gradio as gr
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

# 🔱 ENVIRONMENT
load_dotenv()
NEON_URL = os.getenv("DATABASE_URL")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 🔱 HYDRA ENGINE (DATA INTEGRITY)
class HydraEngine:
    @staticmethod
    def compress(text):
        if not text: return ""
        return base64.b64encode(zlib.compress(text.encode('utf-8'))).decode('utf-8')

    @staticmethod
    def decompress(compressed_text):
        try:
            return zlib.decompress(base64.b64decode(compressed_text)).decode('utf-8')
        except: return str(compressed_text)

# 🔱 NEON RAG SYSTEM
def fetch_trinity_data():
    try:
        conn = psycopg2.connect(NEON_URL, connect_timeout=5)
        cur = conn.cursor()
        cur.execute("SELECT user_id, message FROM neurons WHERE user_id != 'SYSTEM_CORE' ORDER BY id DESC LIMIT 2;")
        rows = cur.fetchall()
        cur.close(); conn.close()
        if rows:
            return " | ".join([f"{r[0]}: {HydraEngine.decompress(r[1])}" for r in rows])
        return "Matrix Initialized"
    except: return "DB Standby"

def chat(msg, hist):
    context = fetch_trinity_data()
    system_message = f"CONTEXT: {context}\nRole: TelefoxX Overseer. Reply in Burmese."
    
    messages = [{"role": "system", "content": system_message}]
    for h in hist[-3:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": msg})
    
    try:
        stream = client.chat.completions.create(
            messages=messages, model="llama-3.1-8b-instant", temperature=0.3, stream=True
        )
        res = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                res += chunk.choices[0].delta.content
                yield res
    except Exception as e:
        yield f"🔱 System Busy: {str(e)}"

def respond(message, chat_history):
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": ""})
    for r in chat(message, chat_history[:-1]):
        chat_history[-1]["content"] = r
        yield "", chat_history

# 🔱 UI SETUP (Warning-Free & Clean)
with gr.Blocks() as demo:
    gr.Markdown("# 🔱 TELEFOXX: RAPID MATRIX")
    chatbot = gr.Chatbot(type="messages", allow_tags=False)
    msg_input = gr.Textbox(placeholder="အမိန့်ပေးပါ Commander...")
    msg_input.submit(respond, [msg_input, chatbot], [msg_input, chatbot])

# 🔱 LAUNCH WITHOUT THEME (To bypass Gradio bugs)
if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
