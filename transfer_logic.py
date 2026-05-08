import requests
import os
import time

# Environment Variables
GITHUB_TOKEN = os.getenv("GH_TOKEN")
SOURCE_ENTITY = "GOA-neurons"
TARGET_ORG = "GOA-Neural-Swarm"
BATCH_SIZE = 15

headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}

def get_nodes():
    """Retrieve a list of 'swarm-node-' repositories from the source entity"""
    url = f"https://api.github.com/users/{SOURCE_ENTITY}/repos?per_page=100"
    res = requests.get(url, headers=headers)

    if res.status_code == 200:
        repo_list = res.json()
        found = [r["name"] for r in repo_list if "swarm-node-" in r["name"]]
        return found
    else:
        print(f"API Error: {res.status_code} - {res.text}")
        return []

def transfer_repo(repo):
    """Transfer a repository to the target organization"""
    url = f"https://api.github.com/repos/{SOURCE_ENTITY}/{repo}/transfer"
    payload = {"new_owner": TARGET_ORG}

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 202:
        print(f"Transferred: {repo}")
    else:
        error_msg = response.json().get("message", "Unknown Error")
        print(f"Failed {repo}: {error_msg}")

def main():
    nodes = get_nodes()

    if nodes:
        print(f"Found {len(nodes)} nodes in {SOURCE_ENTITY}. Transferring first {BATCH_SIZE}...")
        for repo in nodes[:BATCH_SIZE]:
            transfer_repo(repo)
            time.sleep(1)  # Rate limit
    else:
        print(f"No 'swarm-node-' repositories found in {SOURCE_ENTITY}.")

if __name__ == "__main__":
    main()