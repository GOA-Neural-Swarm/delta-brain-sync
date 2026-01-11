import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import psycopg2
import json

# ၁။ လိုအပ်တာတွေ ဇွတ်သွင်း
os.system('pip install -q -U transformers accelerate bitsandbytes psycopg2-binary')

def run_analysis():
    NEON_URL = "postgresql://neondb_owner:npg_QUqg12MzNxnI@ep-long-sound-ahsjjrnk-pooler.c-3.us-east-1.aws.neon.tech/neondb?sslmode=require"
    model_id = "unsloth/llama-3-8b-Instruct-bnb-4bit"
    
    # Brain Loading (GPU)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Database Fetch
    conn = psycopg2.connect(NEON_URL)
    cur = conn.cursor()
    cur.execute("SELECT (data->>'gen')::int, (data->>'bias')::float FROM neurons ORDER BY evolved_at DESC LIMIT 1;")
    row = cur.fetchone()
    
    gen, bias = row[0], row[1]
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a strategic AI core.<|eot_id|><|start_header_id|>user<|end_header_id|>Analyze Gen {gen} (Bias: {bias}).<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    
    output = pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
    thought = output[0]['generated_text'].split('<|assistant|>')[-1].strip()
    
    # AI ရဲ့ အတွေးကို Database ထဲ ဇွတ်ပြန်သိမ်းမယ်
    cur.execute("UPDATE neurons SET data = data || %s WHERE (data->>'gen')::int = %s", 
                (json.dumps({"ai_thought": thought}), gen))
    
    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ Gen {gen} Thought Saved!")

if __name__ == "__main__":
    run_analysis()
  
