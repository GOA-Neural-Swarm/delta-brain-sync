# 🧬 [QUANTUM_EVOLUTION]: Gen_84 Linked
import telemetry_bridge
import os
import time
import requests
import logging
from typing import List
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
GITHUB_TOKEN = os.getenv('GH_TOKEN')
SOURCE_ENTITY = 'GOA-neurons'
TARGET_ORG = 'GOA-Neural-Swarm'
BATCH_SIZE = 15
MAX_RETRIES = 3
headers = {'Authorization': f'token {GITHUB_TOKEN}', 'Accept': 'application/vnd.github.v3+json', 'User-Agent': 'Swarm-Node-Transfer'}

def get_repositories() -> List[str]:
    """
    Retrieve a list of 'swarm-node-' repositories from the source entity.

    Returns:
        List[str]: A list of repository names
    """
    url = f'https://api.github.com/users/{SOURCE_ENTITY}/repos'
    params = {'per_page': 100, 'type': 'all', 'state': 'all'}
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        repositories = response.json()
        return [repo['name'] for repo in repositories if 'swarm-node-' in repo['name']]
    except requests.RequestException as e:
        logging.error(f'API Error: {e}')
        return []

def transfer_repository(repository: str) -> bool:
    """
    Transfer a repository to the target organization.

    Args:
        repository (str): The name of the repository to transfer

    Returns:
        bool: Transfer status
    """
    url = f'https://api.github.com/repos/{SOURCE_ENTITY}/{repository}/transfer'
    payload = {'new_owner': TARGET_ORG, 'team_ids': []}
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        if response.status_code == 202:
            logging.info(f'Transferred: {repository}')
            return True
        else:
            logging.error(f'Failed to transfer {repository}: {response.text}')
            return False
    except requests.RequestException as e:
        logging.error(f'Failed to transfer {repository}: {e}')
        return False

def process_repositories_in_batches(repositories: List[str], batch_size: int) -> List[str]:
    """
    Process a list of repositories in batches.

    Args:
        repositories (List[str]): A list of repository names
        batch_size (int): The size of each batch

    Returns:
        List[str]: A list of failed repository transfers
    """
    failed_transfers = []
    for i in range(0, len(repositories), batch_size):
        batch = repositories[i:i + batch_size]
        for repository in batch:
            if not transfer_repository(repository):
                failed_transfers.append(repository)
            time.sleep(1)
    return failed_transfers

def retry_failed_transfers(failed_transfers: List[str], max_retries: int=MAX_RETRIES) -> None:
    """
    Retry failed repository transfers.

    Args:
        failed_transfers (List[str]): A list of repository names that failed transfer
        max_retries (int): The maximum number of retries. Defaults to MAX_RETRIES.
    """
    for _ in range(max_retries):
        new_failed_transfers = []
        for repository in failed_transfers:
            if not transfer_repository(repository):
                new_failed_transfers.append(repository)
        failed_transfers = new_failed_transfers
        if not failed_transfers:
            break
    if failed_transfers:
        logging.error(f'Failed to transfer {len(failed_transfers)} repositories after {max_retries} retries: {failed_transfers}')

def main() -> None:
    """
    Main function to execute the repository transfer process.
    """
    repositories = get_repositories()
    if repositories:
        logging.info(f'Found {len(repositories)} nodes in {SOURCE_ENTITY}. Transferring in batches of {BATCH_SIZE}...')
        failed_transfers = process_repositories_in_batches(repositories, BATCH_SIZE)
        if failed_transfers:
            retry_failed_transfers(failed_transfers)
    else:
        logging.info(f"No 'swarm-node-' repositories found in {SOURCE_ENTITY}.")
if __name__ == '__main__':
    main()