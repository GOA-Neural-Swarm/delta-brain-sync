import os
import time
import logging
import requests
from typing import List
import telemetry_bridge

GITHUB_TOKEN = os.getenv("GH_TOKEN")
SOURCE_ENTITY = "GOA-neurons"
TARGET_ORG = "GOA-Neural-Swarm"
BATCH_SIZE = 15
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
    "User-Agent": "Swarm-Node-Transfer",
}


class Retry:
    """
    A retry mechanism for handling transient errors.
    """

    def __init__(self, max_retries: int = 3, backoff_factor: float = 1):
        """
        Initialize the retry mechanism.

        Args:
        - max_retries (int): The maximum number of retries.
        - backoff_factor (float): The backoff factor for exponential backoff.
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def retry(self, func, *args, **kwargs):
        """
        Retry a function with exponential backoff.

        Args:
        - func: The function to retry.
        - *args: The positional arguments for the function.
        - **kwargs: The keyword arguments for the function.

        Returns:
        - The result of the function if it succeeds, or raises an exception if all retries fail.
        """
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logging.warning(
                        f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {self.backoff_factor * 2 ** attempt} seconds..."
                    )
                    time.sleep(self.backoff_factor * 2**attempt)
                else:
                    logging.error(f"All retries failed: {str(e)}")
                    raise


def get_nodes() -> List[str]:
    """
    Retrieve a list of 'swarm-node-' repositories from the source entity.

    Returns:
    - A list of repository names.
    """
    url = f"https://api.github.com/users/{SOURCE_ENTITY}/repos?per_page=100"
    params = {"type": "all", "state": "all"}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        repo_list = response.json()
        return [r["name"] for r in repo_list if "swarm-node-" in r["name"]]
    else:
        logging.error(f"API Error: {response.status_code} - {response.text}")
        return []


def transfer_repo(repo: str) -> bool:
    """
    Transfer a repository to the target organization.

    Args:
    - repo (str): The name of the repository to transfer.

    Returns:
    - Transfer status (True if successful, False otherwise).
    """
    url = f"https://api.github.com/repos/{SOURCE_ENTITY}/{repo}/transfer"
    payload = {"new_owner": TARGET_ORG, "team_ids": []}
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 202:
        logging.info(f"Transferred: {repo}")
        return True
    else:
        error_msg = response.json().get("message", "Unknown Error")
        logging.error(f"Failed {repo}: {error_msg}")
        return False


def process_repos_in_batches(repos: List[str], batch_size: int) -> None:
    """
    Process a list of repositories in batches.

    Args:
    - repos (List[str]): A list of repository names.
    - batch_size (int): The size of each batch.
    """
    retry = Retry(max_retries=3, backoff_factor=1)
    for i in range(0, len(repos), batch_size):
        batch = repos[i : i + batch_size]
        for repo in batch:
            if not retry.retry(transfer_repo, repo):
                logging.warning(f"Failed to transfer {repo} after retries")
        time.sleep(1)


def main() -> None:
    """
    The main entry point of the program.
    """
    nodes = get_nodes()
    if nodes:
        logging.info(
            f"Found {len(nodes)} nodes in {SOURCE_ENTITY}. Transferring in batches of {BATCH_SIZE}..."
        )
        process_repos_in_batches(nodes, BATCH_SIZE)
    else:
        logging.info(f"No 'swarm-node-' repositories found in {SOURCE_ENTITY}.")


if __name__ == "__main__":
    main()
