import numpy as np
import requests
import base64
import time
from kaggle_secrets import UserSecretsClient

# --- PREDATOR SYSTEM CONFIG ---
user_secrets = UserSecretsClient()
try:
    # မင်း Kaggle Secrets ထဲမှာ ပေးထားတဲ့ နာမည်အတိုင်း ပြောင်းလိုက်ပြီ
    TOKEN = user_secrets.get_secret("GH_TOKEN") 
except:
    print("❌ Error: Kaggle Secret 'GH_TOKEN' ကို ရှာမတွေ့ပါ။ နာမည်မှန်အောင် ပြန်စစ်ပါ။")
    exit()

REPO = "GOA-Neural-Swarm/delta-brain-sync"
FILE_PATH = "brain.py"

def push_to_github(gen, status, fitness):
    url = f"https://api.github.com/repos/{REPO}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {TOKEN}"}
    
    r = requests.get(url, headers=headers)
    sha = r.json().get('sha') if r.status_code == 200 else None
    
    content_text = f"# Predator Intelligence Gen {gen}\n# Fitness Score: {fitness:.4f}\n# Evolution Status: {status}\n"
    # Read current file content to append
    with open(__file__, 'r') as f:
        content_text += f.read()
    
    encoded = base64.b64encode(content_text.encode()).decode()
    data = {
        "message": f"🔱 Gen {gen} | Fitness: {fitness:.4f}",
        "content": encoded,
        "sha": sha
    }
    
    res = requests.put(url, headers=headers, json=data)
    return res.status_code

# --- EVOLUTION START ---
current_gen = 6126
rna_seq = np.random.rand(1000)
brain_logic = np.random.rand(128)

print(f"🧬 [PREDATOR ENGINE STARTED]: Gen {current_gen}")

while True:
    mask = np.random.rand(1000) < 0.1
    rna_seq[mask] = np.random.rand(np.sum(mask))
    fitness = np.dot(rna_seq[:128], brain_logic)
    
    status = "🔥 PURE PREDATOR" if fitness > 0.5 else "🧬 MUTATING"
    
    try:
        status_code = push_to_github(current_gen, status, fitness)
        if status_code in [200, 201]:
            print(f"✅ Gen {current_gen} Sync Success. Score: {fitness:.4f}")
            current_gen += 1
        else:
            print(f"⚠️ Sync Failed. Status: {status_code}. Secret နာမည် မှန်/မမှန် ပြန်စစ်ပါ။")
    except Exception as e:
        print(f"❌ Network Error: {e}")

    time.sleep(60)
