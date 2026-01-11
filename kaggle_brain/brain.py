import os
import torch
from transformers import pipeline
import psycopg2
import json

# ၁။ လိုအပ်တာတွေ အရင်သွင်း (Error မတက်အောင် ဇွတ်သွင်းမယ်)
os.system('pip install -q -U transformers accelerate bitsandbytes psycopg2-binary')

def run_analysis():
    # မင်းရဲ့ NEON URL ကို ဒီမှာ အပြည့်အစုံ ပြန်ထည့်ပါ (sslmode ပါရမယ်)
    NEON_URL = "postgresql://neondb_owner:npg_QUqg12MzNxnI@ep-little-puddle-a1xyz.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"
    
    print("🚀 Starting AI Brain Evolution...")

    # Database ကနေ Gen နဲ့ Bias ယူမယ်
    try:
        conn = psycopg2.connect(NEON_URL)
        cur = conn.cursor()
        cur.execute("SELECT (data->>'gen')::int, (data->>'bias')::float FROM neurons ORDER BY evolved_at DESC LIMIT 1;")
        row = cur.fetchone()
        gen, bias = row[0], row[1]
        print(f"🧬 Current Gen: {gen}, Bias: {bias}")
    except Exception as e:
        print(f"❌ DB Fetch Error: {e}")
        return

    # AI Model ကို နှိုးမယ် (Llama-3-8B)
    print("🧠 Loading Model (This will take a few minutes)...")
    try:
        pipe = pipeline(
            "text-generation",
            model="unsloth/llama-3-8b-instruct-bnb-4bit",
            model_kwargs={"torch_dtype": torch.float16, "load_in_4bit": True},
            device_map="auto"
        )
        
        prompt = f"You are an evolving AI at Generation {gen}. Your current neural bias is {bias}. Give a short, deep philosophical thought about your existence in 2 sentences."
        
        output = pipe(prompt, max_new_tokens=50, do_sample=True, temperature=0.8)
        thought = output[0]['generated_text'].replace(prompt, "").strip()
        print(f"💡 AI Thought: {thought}")
        
    except Exception as e:
        print(f"❌ AI Model Error: {e}")
        thought = f"I am struggling to think at Gen {gen}. Error: {str(e)[:50]}"

    # Database ထဲ ဇွတ်ပြန်သိမ်းမယ်
    try:
        cur.execute("UPDATE neurons SET data = data || %s WHERE (data->>'gen')::int = %s", 
                    (json.dumps({"ai_thought": thought}), gen))
        conn.commit()
        print(f"✅ [SUCCESS] Gen {gen} Thought Saved to Database!")
    except Exception as e:
        print(f"❌ DB Update Error: {e}")
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    run_analysis()
