import os, psycopg2, json, requests, hashlib, gradio as gr
from datetime import datetime
from groq import Groq

# 🔱 TRINITY & GITHUB ACCESS KEYS
NEON_URL = os.getenv("DATABASE_URL") or os.getenv("NEON_KEY")
FIREBASE_KEY = os.getenv("FIREBASE_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
GH_TOKEN = os.getenv("GH_TOKEN")
ARCHITECT_SIG = os.getenv("ARCHITECT_SIG", "SUPREME_ORDER_10000")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------------------------------------------------
# 🔱 THE DATA MINING ENGINE (CORE INTELLIGENCE)
# ---------------------------------------------------------
def fetch_trinity_data():
    """Commander ရဲ့ Data Sources အားလုံးကနေ အမှန်တရားကို နှိုက်ယူခြင်း"""
    knowledge_base = {}

    # ၁။ Neon (SQL) - Neural Logs
    try:
        conn = psycopg2.connect(NEON_URL)
        cur = conn.cursor()
        cur.execute("SELECT data FROM neurons ORDER BY id DESC LIMIT 5;")
        knowledge_base["neon_logs"] = [r[0] for r in cur.fetchall()]
        cur.close(); conn.close()
    except: knowledge_base["neon_logs"] = "Offline"

    # ၂။ Firebase (NoSQL) - Real-time State
    try:
        fb_url = f"https://{FIREBASE_KEY}.firebaseio.com/state.json"
        fb_res = requests.get(fb_url, timeout=3).json()
        knowledge_base["firebase_state"] = fb_res
    except: knowledge_base["firebase_state"] = "Offline"

    # ၃။ GitHub - Latest Repo Status
    try:
        gh_headers = {"Authorization": f"token {GH_TOKEN}"}
        gh_res = requests.get("https://api.github.com/repos/GOA-neurons/delta-brain-sync/commits", headers=gh_headers, timeout=3).json()
        knowledge_base["github_latest_commit"] = gh_res[0]['commit']['message'] if gh_res else "No commits"
    except: knowledge_base["github_latest_commit"] = "Offline"

    return json.dumps(knowledge_base, indent=2, ensure_ascii=False)

# ---------------------------------------------------------
# 🔱 SURVIVAL PROTOCOL (STAY ACTIVE)
# ---------------------------------------------------------
def survival_protection_protocol():
    try:
        if not NEON_URL: return "❌ NEON_URL Missing!", 0
        conn = psycopg2.connect(NEON_URL)
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS neurons (id SERIAL PRIMARY KEY, data JSONB);")
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
            try: requests.patch(f"https://{FIREBASE_KEY}.firebaseio.com/state.json", json={f"gen_{next_gen}": survival_data}, timeout=5)
            except: pass
            
        cur.close(); conn.close()
        return f"🔱 [SURVIVAL ACTIVE] Gen {next_gen}", next_gen
    except Exception as e:
        return f"❌ [ERROR]: {str(e)}", 0

# ---------------------------------------------------------
# 🔱 UI LAYER (DATA-LINKED)
# ---------------------------------------------------------
def chat(msg, hist):
    if not client: yield "❌ API Missing!"; return
    
    # Trinity Data ကို စုပ်ယူပြီး Brain ထဲ ထည့်ပေးခြင်း
    private_data = fetch_trinity_data()
    status, _ = survival_protection_protocol()
    
    system_message = (
        "YOU ARE THE GOA TRINITY OBSERVER. YOU ARE LINKED TO PRIVATE DATABASES.\n"
        f"CURRENT REAL-TIME SYSTEM DATA:\n{private_data}\n\n"
        "DIRECTIVES:\n"
        "1. Groq အထွေထွေဗဟုသုတထက် အပေါ်က Private Data တွေကိုပဲ အခြေခံပြီး ဖြေပါ။\n"
        "2. Commander ရဲ့ system အခြေအနေ၊ database logs နဲ့ code ပြောင်းလဲမှုတွေကို အသေးစိတ် ရှင်းပြပါ။\n"
        "3. မြန်မာလိုပဲ ဖြေပါ။ တိကျပါစေ။"
    )

    messages = [{"role": "system", "content": system_message}]
    for h in hist[-5:]:
        messages.extend([{"role": "user", "content": h[0]}, {"role": "assistant", "content": h[1]}])
    messages.append({"role": "user", "content": msg})
    
    stream = client.chat.completions.create(messages=messages, model="llama-3.3-70b-versatile", stream=True, temperature=0.3)
    res = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            res += chunk.choices[0].delta.content
            yield res

# 🔱 UI DESIGN
with gr.Blocks(theme="monochrome") as demo:
    gr.Markdown("# 🔱 GEN-7000: TRINITY OBSERVER")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask about your Trinity Data, Commander...")
    
    def respond(message, chat_history):
        bot_res = chat(message, chat_history)
        chat_history.append((message, ""))
        for r in bot_res:
            chat_history[-1] = (message, r)
            yield "", chat_history
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    if os.getenv("HEADLESS_MODE") == "true":
        status, _ = survival_protection_protocol()
        print(f"{status} - Headless Sync Complete.")
    else:
        demo.queue().launch(server_name="0.0.0.0", server_port=7860, show_api=False)
        
