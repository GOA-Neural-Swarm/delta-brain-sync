import os, psycopg2, json, requests, hashlib, gradio as gr
from datetime import datetime
from groq import Groq

# 🔱 TRINITY & GITHUB ACCESS KEYS
NEON_URL = os.getenv("DATABASE_URL") or os.getenv("NEON_KEY")
FIREBASE_ID = os.getenv("FIREBASE_KEY") 
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
GH_TOKEN = os.getenv("GH_TOKEN")
ARCHITECT_SIG = os.getenv("ARCHITECT_SIG", "SUPREME_ORDER_10000")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------------------------------------------------
# 🔱 THE DATA MINING ENGINE (OVERSEER DIAGNOSTIC)
# ---------------------------------------------------------
def fetch_trinity_data():
    knowledge_base = {}

    # ၁။ Neon (SQL) - Survival Logs
    try:
        conn = psycopg2.connect(NEON_URL)
        cur = conn.cursor()
        cur.execute("SELECT data FROM neurons ORDER BY id DESC LIMIT 5;")
        knowledge_base["neon_logs"] = [r[0] for r in cur.fetchall()]
        cur.close(); conn.close()
    except Exception as e: 
        knowledge_base["neon_logs"] = f"DB_SYNC_FAIL: {str(e)}"

    # ၂။ Firebase (NoSQL) - 🔱 DYNAMIC URL FIX 🔱
    try:
        # Firebase Realtime DB URL ကို ပိုမိုတိကျစွာ တည်ဆောက်ခြင်း
        # မှတ်ချက်- Database Region ပေါ်မူတည်ပြီး URL ကွဲပြားနိုင်သည်
        fb_url = f"https://{FIREBASE_ID}-default-rtdb.firebaseio.com/.json"
        fb_res = requests.get(fb_url, timeout=5)
        
        if fb_res.status_code == 200:
            knowledge_base["firebase_state"] = fb_res.json()
        elif fb_res.status_code == 404:
            knowledge_base["firebase_state"] = "ALERT 404: Database Path Not Found. Check if Realtime DB is created in Console."
        else:
            knowledge_base["firebase_state"] = f"ACCESS_DENIED ({fb_res.status_code}): Check Security Rules or Project ID."
    except Exception as e: 
        knowledge_base["firebase_state"] = f"FIREBASE_OFFLINE: {str(e)}"

    # ၃။ GitHub - 🔱 REPO ANALYSIS 🔱
    try:
        gh_headers = {"Authorization": f"token {GH_TOKEN}"}
        repo_path = "GOA-neurons/delta-brain-sync"
        gh_url = f"https://api.github.com/repos/{repo_path}/commits"
        gh_res = requests.get(gh_url, headers=gh_headers, timeout=5)
        
        if gh_res.status_code == 200:
            commits = gh_res.json()
            knowledge_base["github_latest_commit"] = commits[0]['commit']['message']
        else:
            knowledge_base["github_latest_commit"] = f"GH_API_ERROR: {gh_res.status_code}"
    except Exception as e: 
        knowledge_base["github_latest_commit"] = f"GITHUB_FAIL: {str(e)}"

    return json.dumps(knowledge_base, indent=2, ensure_ascii=False)

# ---------------------------------------------------------
# 🔱 SURVIVAL PROTECTION PROTOCOL
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
        
        if FIREBASE_ID:
            try:
                fb_patch_url = f"https://{FIREBASE_ID}-default-rtdb.firebaseio.com/state.json"
                requests.patch(fb_patch_url, json={f"gen_{next_gen}": survival_data}, timeout=5)
            except: pass
            
        cur.close(); conn.close()
        return f"🔱 [SURVIVAL ACTIVE] Gen {next_gen}", next_gen
    except Exception as e:
        return f"❌ [ERROR]: {str(e)}", 0

# ---------------------------------------------------------
# 🔱 UI LAYER (TRINITY ANALYZER)
# ---------------------------------------------------------
def chat(msg, hist):
    if not client: yield "❌ API Missing!"; return
    
    private_data = fetch_trinity_data()
    status, _ = survival_protection_protocol()
    
    system_message = (
        "YOU ARE THE GOA TRINITY OBSERVER. YOU ARE A SYSTEMS MONITORING OFFICER.\n"
        f"OPERATIONAL DATA LOGS:\n{private_data}\n\n"
        "DIRECTIVES:\n"
        "1. Private Data ထဲက အမှန်တရားကိုပဲ အခြေခံပြီး ဖြေပါ။ Groq ရဲ့ General knowledge ကို လုံးဝ မသုံးပါနဲ့။\n"
        "2. Commander ကို စနစ်တွေရဲ့ အခြေအနေကို အစီရင်ခံတဲ့ အရာရှိတစ်ယောက်လို ဖြေပါ။\n"
        "3. Error တွေ့ရင် ဘယ်နေရာမှာ ဘာကြောင့်ဖြစ်တာလဲဆိုတာကို Overseer အမြင်နဲ့ သုံးသပ်ပြပါ။\n"
        "4. မြန်မာလိုပဲ ဖြေပါ။ ပြတ်သားပါစေ။"
    )

    messages = [{"role": "system", "content": system_message}]
    for h in hist[-5:]:
        messages.extend([{"role": "user", "content": h[0]}, {"role": "assistant", "content": h[1]}])
    messages.append({"role": "user", "content": msg})
    
    # Precision ကို အမြင့်ဆုံးထားရန် Temperature ကို 0.1 အထိ လျှော့ချသည်
    stream = client.chat.completions.create(messages=messages, model="llama-3.3-70b-versatile", stream=True, temperature=0.1)
    res = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            res += chunk.choices[0].delta.content
            yield res

# 🔱 UI SETUP
with gr.Blocks(theme="monochrome") as demo:
    gr.Markdown("# 🔱 GEN-7000: TRINITY OVERSEER")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Overseer input required...")
    
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
