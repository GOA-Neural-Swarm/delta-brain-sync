import requests
import os
import time

# Environment Variables
GITHUB_TOKEN = os.getenv("GH_TOKEN")
SOURCE_ENTITY = "GOA-neurons"
TARGET_ORG = "GOA-Neural-Swarm"
BATCH_SIZE = 15

# Headers for GitHub API requests
headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
    "User-Agent": "Swarm-Node-Transfer",
}


def get_nodes():
    """
    Retrieve a list of 'swarm-node-' repositories from the source entity

    Returns:
        list: A list of repository names
    """
    url = f"https://api.github.com/users/{SOURCE_ENTITY}/repos?per_page=100"
    params = {"type": "all", "state": "all"}
    res = requests.get(url, headers=headers, params=params)

    if res.status_code == 200:
        repo_list = res.json()
        # Filter repositories by name
        found = [r["name"] for r in repo_list if "swarm-node-" in r["name"]]
        return found
    else:
        # Handle API errors
        print(f"API Error: {res.status_code} - {res.text}")
        return []


def transfer_repo(repo):
    """
    Transfer a repository to the target organization

    Args:
        repo (str): The name of the repository to transfer
    """
    url = f"https://api.github.com/repos/{SOURCE_ENTITY}/{repo}/transfer"
    # Set the new owner for the repository
    payload = {"new_owner": TARGET_ORG, "team_ids": []}

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 202:
        # Log successful transfer
        print(f"Transferred: {repo}")
    else:
        # Log transfer failure
        error_msg = response.json().get("message", "Unknown Error")
        print(f"Failed {repo}: {error_msg}")


def main():
    # Get the list of 'swarm-node-' repositories
    nodes = get_nodes()

    if nodes:
        # Log the number of nodes found
        print(
            f"Found {len(nodes)} nodes in {SOURCE_ENTITY}. Transferring first {BATCH_SIZE}..."
        )
        # Transfer the first BATCH_SIZE repositories
        for repo in nodes[:BATCH_SIZE]:
            transfer_repo(repo)
            # Pause for 1 second to avoid rate limiting
            time.sleep(1)
    else:
        # Log no repositories found
        print(f"No 'swarm-node-' repositories found in {SOURCE_ENTITY}.")


if __name__ == "__main__":
    main()
