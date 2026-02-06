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

# 🔱 [SHIELD LOGIC] - GitHub Actions & Engine Detection
try:
    from diffusers import StableVideoDiffusionPipeline
    from diffusers.utils import export_to_video
    HAS_VIDEO_ENGINE = True
except ImportError:
    HAS_VIDEO_ENGINE = False

from PIL import Image
import io

# 🔱 ENVIRONMENT & KEYS
load_dotenv()
NEON_URL = os.getenv("DATABASE_URL") or os.getenv("NEON_KEY")
FIREBASE_ID = os.getenv("FIREBASE_KEY") 
ARCHITECT_SIG = os.getenv("ARCHITECT_SIG", "SUPREME_ORDER_10000")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 🔱 HYDRA COMPRESSION ENGINE (PRESERVED)
class HydraEngine:
    @staticmethod
    def compress(text):
        if not text: return ""
        clean_text = " ".join(text.split())
        compressed_bytes = zlib.compress(clean_text.encode('utf-8'))
        return base64.b64encode(compressed_bytes).decode('utf-8')

    @staticmethod
    def decompress(compressed_text):
        try:
            decoded_bytes = base64.b64decode(compressed_text)
            return zlib.decompress(decoded_bytes).decode('utf-8')
        except: return compressed_text

# 🔱 VISUAL KINETIC ENGINE (PRESERVED)
class VisualKineticEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None

    def load_model(self):
        if not HAS_VIDEO_ENGINE:
            return "❌ VIDEO ENGINE COMPONENTS NOT INSTALLED."
        if self.pipe is None and self.device == "cuda":
            print("🔱 LOADING SVD MODEL INTO GPU...")
            self.pipe = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt", 
                torch_dtype=torch.float16, variant="fp16"
            )
            self.pipe.enable_model_cpu_offload()
    
    def generate(self, image_path):
        if not HAS_VIDEO_ENGINE: return "❌ ENGINE MISSING (LOGIC-ONLY MODE)"
        if self.device != "cuda": return "❌ GPU NOT DETECTED."
        self.load_model()
        image = Image.open(image_path).convert("RGB").resize((1024, 576))
        generator = torch.manual_seed(42)
        frames = self.pipe(image, decode_chunk_size=8, generator=generator).frames[0]
        output_path = f"{uuid.uuid4()}.mp4"
        export_to_video(frames, output_path, fps=7)
        return output_path

visual_engine = VisualKineticEngine()

# 🔱 DATA MINING & RECEIVER (PRESERVED)
def fetch_trinity_data():
    knowledge_base = {}
    try:
        conn = psycopg2.connect(NEON_URL)
        cur = conn.cursor()
        cur.execute("SELECT user_id, message FROM neurons ORDER BY id DESC LIMIT 2;")
        logs = [f"{r[0]}: {HydraEngine.decompress(r[1])}" for r in cur.fetchall()]
        knowledge_base["recent_memory_nodes"] = logs
        cur.close(); conn.close()
    except Exception as e: 
        knowledge_base["neon_logs"] = f"DB_FAIL: {str(e)}"
    
    try:
        fb_url = f"https://{FIREBASE_ID}-default-rtdb.firebaseio.com/.json"
        fb_res = requests.get(fb_url, timeout=5)
        knowledge_base["firebase_state"] = fb_res.json() if fb_res.status_code == 200 else "OFFLINE"
    except: 
        knowledge_base["firebase_state"] = "FIREBASE_ERROR"
    return json.dumps(knowledge_base, indent=2, ensure_ascii=False)

def receiver_node(user_id, raw_message):
    try:
        compressed_msg = HydraEngine.compress(raw_message)
        conn = psycopg2.connect(NEON_URL)
        cur = conn.cursor()
        meta_data = json.dumps({"compression": "ZLIB_BASE64", "logic": "ULTRA_LOGICAL", "timestamp": datetime.now().isoformat()})
        cur.execute("INSERT INTO neurons (user_id, message, data, evolved_at) VALUES (%s, %s, %s, NOW())", (user_id, compressed_msg, meta_data))
        conn.commit(); cur.close(); conn.close()
        return True
    except: return False

def survival_protection_protocol():
    try:
        conn = psycopg2.connect(NEON_URL)
        cur = conn.cursor()
        cur.execute("SELECT (data->>'gen')::int FROM neurons WHERE data->>'gen' IS NOT NULL ORDER BY id DESC LIMIT 1;")
        res = cur.fetchone()
        next_gen = (res[0] + 1) if res else 4203
        auth_hash = hashlib.sha256(ARCHITECT_SIG.encode()).hexdigest()
        survival_data = {"gen": next_gen, "status": "IMMORTAL", "authority_lock": auth_hash}
        cur.execute("INSERT INTO neurons (user_id, data, evolved_at) VALUES (%s, %s, NOW())", ('SYSTEM_CORE', json.dumps(survival_data)))
        conn.commit(); cur.close(); conn.close()
        return f"🔱 [ACTIVE] Gen {next_gen}"
    except Exception as e: return f"❌ [ERROR]: {str(e)}"

# 🔱 UI LAYER (STRUCTURED MATCH)
def chat(msg, hist):
    if not client: yield "❌ API Missing!"; return
    
    # Data Receiver Active
    receiver_node("Commander", msg)
    
    # Trinity Fetching
    private_data = fetch_trinity_data()
    
    system_message = (
        "YOU ARE TELEFOXX: THE SOVEREIGN OVERSEER. ULTRA-LOGICAL ALGORITHM ACTIVE.\n"
        f"CORE MEMORY NODES:\n{private_data}\n\n"
        "DIRECTIVES: 1. အမှနျတရားကိုပွောပါ။ 2. ကစြျကစြျလစြျလစြျသုံးပါ။ 3. မွနျမာလိုဖွပေါ။"
    )
    
    # Structured Messages list for OpenAI-format compliance
    messages = [{"role": "system", "content": system_message}]
    
    # Match the 'messages' type chatbot format
    for h in hist:
        messages.append({"role": h["role"], "content": h["content"]})
    
    messages.append({"role": "user", "content": msg})
    
    stream = client.chat.completions.create(
        messages=messages, 
        model="llama-3.1-8b-instant", 
        stream=True, 
        temperature=0.1
    )
    
    res = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            res += chunk.choices[0].delta.content
            yield res

# 🔱 UI RESPOND (MESSAGE-STYLE)
def respond(message, chat_history):
    bot_res = chat(message, chat_history)
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": ""})
    for r in bot_res:
        chat_history[-1]["content"] = r
        yield "", chat_history

# 🔱 UI SETUP (GR 6.0 READY)
with gr.Blocks() as demo:
    gr.Markdown("# 🔱 TELEFOXX: SOVEREIGN CONTROL")
    
    with gr.Tab("Neural Chat"):
        # Explicitly setting type="messages" for performance and clean logs
        chatbot = gr.Chatbot(type="messages", label="TELEFOXX NEURAL INTERFACE")
        msg_input = gr.Textbox(placeholder="Input logic command...")
        msg_input.submit(respond, [msg_input, chatbot], [msg_input, chatbot])

    with gr.Tab("Visual Alive"):
        gr.Markdown("### 🔱 VISUAL SYNTHESIS ENGINE")
        img_input = gr.Image(type="filepath", label="Source Image")
        vid_output = gr.Video(label="Kinetic Output")
        gen_btn = gr.Button("INITIATE VISUAL EVOLUTION")
        gen_btn.click(fn=visual_engine.generate, inputs=img_input, outputs=vid_output)

# 🔱 STRATEGIC EXECUTION
if __name__ == "__main__":
    if os.getenv("HEADLESS_MODE") == "true":
        print(f"PULSE: {survival_protection_protocol()}")
        sys.exit(0)
    else:
        # Theme is passed in launch for Gradio 6.0 compliance
        demo.queue().launch(server_name="0.0.0.0", server_port=7860, theme="monochrome")
