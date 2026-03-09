import requests
import os
import time

# GitHub Secrets
GITHUB_TOKEN = os.getenv("MY_GH_TOKEN") # Workflow ထဲက နာမည်နဲ့ ကိုက်အောင် ပြင်ထားသည်
SOURCE_ORG = "GOA-neurons"
TARGET_ORG = "GOA-Neural-Swarm"
BATCH_SIZE = 10

headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

def get_nodes():
    # 🔱 ပြင်ဆင်ချက်: /users/ အစား /orgs/ ကို ပြောင်းသုံးထားသည်
    url = f"https://api.github.com/orgs/{SOURCE_ORG}/repos?per_page=100&type=all"
    res = requests.get(url, headers=headers)
    
    if res.status_code == 200:
        repo_list = res.json()
        # "swarm-node-" ပါတဲ့ repo နာမည်တွေကို ယူမယ်
        return [r['name'] for r in repo_list if "swarm-node-" in r['name']]
    else:
        print(f"❌ API Error: {res.status_code} - {res.text}")
        return []

nodes = get_nodes()

if nodes:
    print(f"📦 Found {len(nodes)} nodes. Processing first {BATCH_SIZE}...")
    for repo in nodes[:BATCH_SIZE]:
        # Transfer endpoint
        url = f"https://api.github.com/repos/{SOURCE_ORG}/{repo}/transfer"
        payload = {"new_owner": TARGET_ORG}
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 202:
            print(f"✅ Transferred: {repo}")
        else:
            print(f"⚠️ Failed {repo}: {response.json().get('message')}")
        
        time.sleep(1) # API Rate limit မထိအောင် ခဏနားခြင်း
else:
    print(f"✨ No nodes found in {SOURCE_ORG} to transfer.")
