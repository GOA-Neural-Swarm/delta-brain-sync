# 🧬 [QUANTUM_EVOLUTION]: Gen_22 Linked
import telemetry_bridge
import os
import time
import requests
import logging
from typing import List
GITHUB_TOKEN = os.getenv('GH_TOKEN')
SOURCE_ENTITY = 'GOA-neurons'
TARGET_ORG = 'GOA-Neural-Swarm'
BATCH_SIZE = 15
headers = {'Authorization': f'token {GITHUB_TOKEN}', 'Accept': 'application/vnd.github.v3+json', 'User-Agent': 'Swarm-Node-Transfer'}
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_repositories() -> List[str]:
    """
    Retrieve a list of 'swarm-node-' repositories from the source entity

    Returns:
        List[str]: A list of repository names
    """
    url = f'https://api.github.com/users/{SOURCE_ENTITY}/repos?per_page=100'
    params = {'type': 'all', 'state': 'all'}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        repositories = response.json()
        return [repo['name'] for repo in repositories if 'swarm-node-' in repo['name']]
    else:
        logging.error(f'API Error: {response.status_code} - {response.text}')
        return []

def transfer_repository(repository: str) -> bool:
    """
    Transfer a repository to the target organization

    Args:
        repository (str): The name of the repository to transfer

    Returns:
        bool: Transfer status
    """
    url = f'https://api.github.com/repos/{SOURCE_ENTITY}/{repository}/transfer'
    payload = {'new_owner': TARGET_ORG, 'team_ids': []}
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 202:
        logging.info(f'Transferred: {repository}')
        return True
    else:
        error_message = response.json().get('message', 'Unknown Error')
        logging.error(f'Failed {repository}: {error_message}')
        return False

def process_repositories_in_batches(repositories: List[str], batch_size: int) -> List[str]:
    """
    Process a list of repositories in batches

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

def retry_failed_transfers(failed_transfers: List[str], max_retries: int=3) -> None:
    """
    Retry failed repository transfers

    Args:
        failed_transfers (List[str]): A list of repository names that failed transfer
        max_retries (int): The maximum number of retries. Defaults to 3.
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