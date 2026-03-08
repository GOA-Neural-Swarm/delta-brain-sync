import requests
import os
import time

# GitHub Secrets ကနေ Token ကို လှမ်းယူမယ်
GITHUB_TOKEN = os.getenv("GH_TOKEN")
SOURCE_USER = "GOA-neurons"
TARGET_ORG = "GOA-Neural-Swarm"
BATCH_SIZE = 10

headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

def get_nodes():
    url = f"https://api.github.com/users/{SOURCE_USER}/repos?per_page=100"
    res = requests.get(url, headers=headers)
    if res.status_code == 200:
        return [r['name'] for r in res.json() if "swarm-node-" in r['name']]
    return []

nodes = get_nodes()
if nodes:
    print(f"📦 Found {len(nodes)} nodes. Processing first {BATCH_SIZE}...")
    for repo in nodes[:BATCH_SIZE]:
        url = f"https://api.github.com/repos/{SOURCE_USER}/{repo}/transfer"
        requests.post(url, headers=headers, json={"new_owner": TARGET_ORG})
        print(f"✅ Transferred: {repo}")
        time.sleep(1)
else:
    print("✨ No nodes found to transfer.")
