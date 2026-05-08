import requests
import os
import time
from typing import List
import logging

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

# Logger configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_nodes() -> List[str]:
    """
    Retrieve a list of 'swarm-node-' repositories from the source entity

    Returns:
        List[str]: A list of repository names
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
        logging.error(f"API Error: {res.status_code} - {res.text}")
        return []


def transfer_repo(repo: str) -> bool:
    """
    Transfer a repository to the target organization

    Args:
        repo (str): The name of the repository to transfer

    Returns:
        bool: Transfer status
    """
    url = f"https://api.github.com/repos/{SOURCE_ENTITY}/{repo}/transfer"
    # Set the new owner for the repository
    payload = {"new_owner": TARGET_ORG, "team_ids": []}

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 202:
        # Log successful transfer
        logging.info(f"Transferred: {repo}")
        return True
    else:
        # Log transfer failure
        error_msg = response.json().get("message", "Unknown Error")
        logging.error(f"Failed {repo}: {error_msg}")
        return False


def transfer_repos(repos: List[str]) -> None:
    """
    Transfer a list of repositories to the target organization

    Args:
        repos (List[str]): A list of repository names to transfer
    """
    for repo in repos:
        transfer_repo(repo)
        # Pause for 1 second to avoid rate limiting
        time.sleep(1)


def process_repos_in_batches(repos: List[str], batch_size: int) -> None:
    """
    Process a list of repositories in batches

    Args:
        repos (List[str]): A list of repository names
        batch_size (int): The size of each batch
    """
    failed_transfers = []
    for i in range(0, len(repos), batch_size):
        batch = repos[i : i + batch_size]
        failed_batch_transfers = [repo for repo in batch if not transfer_repo(repo)]
        failed_transfers.extend(failed_batch_transfers)
        # Pause for 1 second to avoid rate limiting
        time.sleep(1)
    if failed_transfers:
        logging.warning(
            f"Failed to transfer {len(failed_transfers)} repositories: {failed_transfers}"
        )


def main() -> None:
    # Get the list of 'swarm-node-' repositories
    nodes = get_nodes()

    if nodes:
        # Log the number of nodes found
        logging.info(
            f"Found {len(nodes)} nodes in {SOURCE_ENTITY}. Transferring in batches of {BATCH_SIZE}..."
        )
        # Process the repositories in batches
        process_repos_in_batches(nodes, BATCH_SIZE)
    else:
        # Log no repositories found
        logging.info(f"No 'swarm-node-' repositories found in {SOURCE_ENTITY}.")


if __name__ == "__main__":
    main()