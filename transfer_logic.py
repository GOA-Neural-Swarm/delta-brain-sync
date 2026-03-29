import requests
import os
import time

# 🔱 Environment Variables ကို Workflow ထဲကအတိုင်း တိုက်ရိုက်ယူမယ်
GITHUB_TOKEN = os.getenv("GH_TOKEN") 
SOURCE_ENTITY = "GOA-neurons"
TARGET_ORG = "GOA-Neural-Swarm"
BATCH_SIZE = 15

headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

def get_nodes():
    # 🔱 ပြင်ဆင်ချက်: GOA-neurons သည် User ဖြစ်သောကြောင့် /users/ ကို ပြန်သုံးရပါမည်
    url = f"https://api.github.com/users/{SOURCE_ENTITY}/repos?per_page=100"
    res = requests.get(url, headers=headers)
    
    if res.status_code == 200:
        repo_list = res.json()
        # "swarm-node-" ပါဝင်သော repo များကို ရှာဖွေခြင်း
        found = [r['name'] for r in repo_list if "swarm-node-" in r['name']]
        return found
    else:
        # Error တက်ပါက ဘာကြောင့်လဲဆိုတာ အသေးစိတ်ထုတ်ပြမယ်
        print(f"❌ API Error: {res.status_code} - {res.text}")
        return []

nodes = get_nodes()

if nodes:
    print(f"📦 Found {len(nodes)} nodes in {SOURCE_ENTITY}. Transferring first {BATCH_SIZE}...")
    for repo in nodes[:BATCH_SIZE]:
        # Transfer လုပ်မည့် Endpoint (User repo ဖြစ်စေ၊ Org repo ဖြစ်စေ ဤ path သည် အတူတူပင်ဖြစ်သည်)
        url = f"https://api.github.com/repos/{SOURCE_ENTITY}/{repo}/transfer"
        payload = {"new_owner": TARGET_ORG}
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 202:
            print(f"✅ Transferred: {repo}")
        else:
            error_msg = response.json().get('message', 'Unknown Error')
            print(f"⚠️ Failed {repo}: {error_msg}")
        
        time.sleep(1) # Rate limit ရှောင်ရန်
else:
    print(f"✨ No 'swarm-node-' repositories found in {SOURCE_ENTITY}.")
