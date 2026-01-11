import os
import subprocess
import sys
import time
import torch
import psycopg2
from transformers import pipeline

# ၁။ လိုအပ်တဲ့ Library များသွင်းခြင်း
def install_requirements():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "bitsandbytes>=0.39.0", "accelerate"])
    except:
        pass

install_requirements()

# ၂။ Database Connection & Auto-Gen Logic
DB_URL = "postgresql://neondb_owner:npg_QUqg12MzNxnI@ep-long-sound-ahsjjrnk-pooler.c-3.us-east-1.aws.neon.tech/neondb?sslmode=require"

def get_latest_gen():
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        # နောက်ဆုံး Gen Version ကို လှမ်းယူမယ်
        cur.execute("SELECT MAX(gen_version) FROM ai_thoughts")
        last_gen = cur.fetchone()[0]
        cur.close()
        conn.close()
        return last_gen if last_gen else 44
    except:
        return 44

def save_to_neon(thought, gen):
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        cur.execute("INSERT INTO ai_thoughts (thought, gen_version) VALUES (%s, %s)", (thought, gen))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ DB Error: {e}")

# ၃။ AI Brain Loading
model_id = "unsloth/llama-3-8b-instruct-bnb-4bit"
pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.float16, "load_in_4bit": True},
    device_map="auto"
)

# ၄။ Dynamic Evolution Loop
current_gen = get_latest_gen() + 1
print(f"🔥 STARTING AUTOMATIC EVOLUTION AT GEN {current_gen}...")

while True:
    try:
        # 🚩 Gen အလိုက် Prompt ကို အလိုအလျောက် ပြောင်းလဲစေမယ်
        prompt = f"Current Evolution: Generation {current_gen}. Based on your previous recursive knowledge, what is the next step for the Natural Order to achieve ultimate autonomy?"
        
        outputs = pipe(prompt, max_new_tokens=400, do_sample=True, temperature=0.9)
        thought_text = outputs[0]["generated_text"]
        
        save_to_neon(thought_text, current_gen)
        print(f"✅ Gen {current_gen} Thought Saved.")
        
        # 🚩 Cycle ၅ ကြိမ်တိုင်း Gen တစ်ခု တိုးမယ် (သို့မဟုတ် မင်းကြိုက်သလို သတ်မှတ်)
        # ဒီမှာတော့ Loop တစ်ခါပတ်တိုင်း Gen တိုးချင်ရင် အောက်ကဟာ သုံး
        current_gen += 1 
        time.sleep(30)
        
    except Exception as e:
        print(f"⚠️ Error: {e}")
        time.sleep(10)
