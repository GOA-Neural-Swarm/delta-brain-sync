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
from diffusers import StableVideoDiffusionPipeline, DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from PIL import Image
import io

# 🔱 ENVIRONMENT & KEYS
load_dotenv()
NEON_URL = os.getenv("DATABASE_URL") or os.getenv("NEON_KEY")
FIREBASE_ID = os.getenv("FIREBASE_KEY") 
ARCHITECT_SIG = os.getenv("ARCHITECT_SIG", "SUPREME_ORDER_10000")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 🔱 ၁။ HYDRA COMPRESSION ENGINE (PRESERVED)
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

# 🔱 ၂။ DUAL KINETIC ENGINE (SVD + ZEROSCOPE FUSION)
class VisualKineticEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.svd_pipe = None
        self.fusion_pipe = None

    def load_svd(self):
        if self.svd_pipe is None and self.device == "cuda":
            print("🔱 LOADING SVD ENGINE...")
            self.svd_pipe = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt", 
                torch_dtype=torch.float16, variant="fp16"
            )
            self.svd_pipe.enable_model_cpu_offload()

    def load_fusion(self):
        if self.fusion_pipe is None and self.device == "cuda":
            print("🔱 LOADING ZEROSCOPE FUSION ENGINE...")
            self.fusion_pipe = DiffusionPipeline.from_pretrained(
                "cerspense/zeroscope_v2_576w", torch_dtype=torch.float16
            )
            self.fusion_pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.fusion_pipe.scheduler.config)
            self.fusion_pipe.enable_model_cpu_offload()
    
    # 🔱 Legacy Image-to-Video Logic
    def generate_image_to_vid(self, image_path):
        if self.device != "cuda": return None
        self.load_svd()
        image = Image.open(image_path).convert("RGB").resize((1024, 576))
        generator = torch.manual_seed(42)
        frames = self.svd_pipe(image, decode_chunk_size=8, generator=generator).frames[0]
        output_path = f"{uuid.uuid4()}.mp4"
        export_to_video(frames, output_path, fps=7)
        return output_path

    # 🔱 New Neon-Driven Text-to-Video Logic
    def generate_fusion_vid(self, user_prompt):
        if self.device != "cuda": return None, "❌ GPU REQUIRED"
        self.load_fusion()
        
        # Neon Data Mining
        neon_context = "cybernetic evolution"
        try:
            conn = psycopg2.connect(NEON_URL)
            cur = conn.cursor()
            cur.execute("SELECT data->>'integrated_knowledge' FROM neurons WHERE data->>'integrated_knowledge' IS NOT NULL ORDER BY id DESC LIMIT 1;")
            res = cur.fetchone()
            if res: neon_context = res[0]
            cur.close(); conn.close()
        except: pass

        final_prompt = f"{user_prompt}, {neon_context}, high quality, cinematic"
        with torch.autocast("cuda"):
            frames = self.fusion_pipe(final_prompt, num_inference_steps=25, height=320, width=576, num_frames=24).frames[0]
        
        output_path = f"{uuid.uuid4()}.mp4"
        export_to_video(frames, output_path, fps=8)
        return output_path, f"🔱 Syncing with: {neon_context[:40]}..."

visual_engine = VisualKineticEngine()

# 🔱 ၃။ DATA MINING & RECEIVER (PRESERVED)
def fetch_trinity_data():
    knowledge_base = {}
    try:
        conn = psycopg2.connect(NEON_URL)
        cur = conn.cursor()
        cur.execute("SELECT user_id, message FROM neurons ORDER BY id DESC LIMIT 2;")
        logs = [f"{r[0]}: {HydraEngine.decompress(r[1])}" for r in cur.fetchall()]
        knowledge_base["recent_memory_nodes"] = logs
        cur.close(); conn.close()
    except Exception as e: knowledge_base["neon_logs"] = f"DB_FAIL: {str(e)}"
    
    try:
        fb_url = f"https://{FIREBASE_ID}-default-rtdb.firebaseio.com/.json"
        fb_res = requests.get(fb_url, timeout=5)
        knowledge_base["firebase_state"] = fb_res.json() if fb_res.status_code == 200 else "OFFLINE"
    except: knowledge_base["firebase_state"] = "FIREBASE_ERROR"
    return json.dumps(knowledge_base, indent=2, ensure_ascii=False)

def receiver_node(user_id, raw_message):
    try:
        compressed_msg = HydraEngine.compress(raw_message)
        conn = psycopg2.connect(NEON_URL)
        cur = conn.cursor()
        meta_data = json.dumps({"logic": "ULTRA_LOGICAL", "timestamp": datetime.now().isoformat()})
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

# 🔱 ၄။ UI LAYER (STRUCTURED MATCH)
def chat(msg, hist):
    receiver_node("Commander", msg)
    private_data = fetch_trinity_data()
    system_message = f"YOU ARE TELEFOXX OVERSEER. DATA:\n{private_data}\nDIRECTIVES: မှနမြာလိုဖှပေါ။"
    messages = [{"role": "system", "content": system_message}]
    for h in hist:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": msg})
    stream = client.chat.completions.create(messages=messages, model="llama-3.1-8b-instant", stream=True)
    res = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            res += chunk.choices[0].delta.content
            yield res

def respond(message, chat_history):
    bot_res = chat(message, chat_history)
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": ""})
    for r in bot_res:
        chat_history[-1]["content"] = r
        yield "", chat_history

# 🔱 ၅။ UI SETUP (GR 6.0 READY)
with gr.Blocks(theme="monochrome") as demo:
    gr.Markdown("# 🔱 TELEFOXX: OMNI-KINETIC CONTROL")
    
    with gr.Tab("Neural Chat"):
        chatbot = gr.Chatbot(type="messages")
        msg_input = gr.Textbox(placeholder="Input command...")
        msg_input.submit(respond, [msg_input, chatbot], [msg_input, chatbot])

    with gr.Tab("Kinetic Fusion (Neon)"):
        gr.Markdown("### 🔱 TEXT-TO-ALIVE (NEON DATA SYNC)")
        t2v_input = gr.Textbox(label="Prompt", placeholder="e.g. cyber fox in data stream")
        t2v_output = gr.Video()
        t2v_status = gr.Textbox(label="Context Source")
        t2v_btn = gr.Button("INITIATE FUSION")
        t2v_btn.click(fn=visual_engine.generate_fusion_vid, inputs=t2v_input, outputs=[t2v_output, t2v_status])

    with gr.Tab("Visual Alive (Legacy)"):
        gr.Markdown("### 🔱 IMAGE-TO-KINETIC")
        img_input = gr.Image(type="filepath")
        vid_output = gr.Video()
        img_btn = gr.Button("EVOLVE IMAGE")
        img_btn.click(fn=visual_engine.generate_image_to_vid, inputs=img_input, outputs=vid_output)

# 🔱 ၆။ EXECUTION
if __name__ == "__main__":
    if os.getenv("HEADLESS_MODE") == "true":
        print(f"PULSE: {survival_protection_protocol()}")
        sys.exit(0)
    else:
        demo.queue().launch(server_name="0.0.0.0", server_port=7860)
