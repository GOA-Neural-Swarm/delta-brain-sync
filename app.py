import telemetry_bridge
import os
import sys
import zlib
import base64
import json
import time
import subprocess
import asyncio
import backoff
import re
import shutil
import git
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from huggingface_hub import HfApi
from dotenv import load_dotenv
from groq import Groq
from google import genai
from google.genai import types
from datasets import load_dataset

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def retry_async_operation(operation, *args, **kwargs):
    return await operation(*args, **kwargs)
load_dotenv()
NEON_DB_URL = os.environ.get('NEON_DB_URL') or os.environ.get('DATABASE_URL')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
HF_TOKEN = os.environ.get('HF_TOKEN')
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
REPO_URL = os.environ.get('REPO_URL') or 'GOA-Neural-Swarm/delta-brain-sync'
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
REPO_PATH = './repo_sync'
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
client = None
if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        print('[GEMINI]: Sovereign Architect (V2) Brain Initialized.')
    except Exception as e:
        print(f'[GEMINI]: Initialization Failed: {e}')
else:
    print('[GEMINI]: API Key missing. Architect mode disabled.')

def generate_brain_evolution(prompt_text):
    if not client:
        return None
    response = client.models.generate_content(model='gemini-2.0-flash', contents=prompt_text)
    return response.text

async def audit_code_integrity(code_content, original_intent):
    if not client:
        print(' [GEMINI AUDIT]: API Key missing. Skipping code audit.')
        return (True, 'Audit skipped due to missing API key.')
    audit_prompt = f"""You are an expert Python code auditor. Your task is to review a piece of generated Python code and determine if it adheres to the original intent and is syntactically correct. You should also check for potential bugs or logical flaws.\nOriginal Intent: {original_intent}\nGenerated Code:\n{code_content}\nProvide a concise 'PASS' or 'FAIL' verdict, followed by a brief explanation. If 'FAIL', suggest specific improvements. Example: 'PASS: Code is clean and matches intent.' or 'FAIL: Missing import statement for 'requests'." \n"""
    try:
        audit_response = generate_brain_evolution(audit_prompt)
        if audit_response and 'PASS' in audit_response.upper():
            print(f' [GEMINI AUDIT]: Code integrity check PASSED. {audit_response}')
            return (True, audit_response)
        else:
            print(f' [GEMINI AUDIT]: Code integrity check FAILED. {audit_response}')
            return (False, audit_response)
    except Exception as e:
        print(f' [GEMINI AUDIT ERROR]: {e}')
        return (False, f'Audit failed due to internal error: {e}')

class HydraEngine:

    @staticmethod
    def compress(data):
        if not data:
            return ''
        return base64.b64encode(zlib.compress(data.encode('utf-8'), level=9)).decode('utf-8')

    @staticmethod
    def decompress(compressed_data):
        try:
            return zlib.decompress(base64.b64decode(compressed_data)).decode('utf-8')
        except:
            return str(compressed_data)

class TelefoxXAGI:

    def __init__(self):
        self._groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
        self.engine = self._create_neon_engine()
        self.sb = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None
        self.models = ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant']
        self.avg_error = 0.0
        self.last_error_log = 'None'
        self.current_gen = 1

    def self_coding_engine_internal(self, raw_content):
        blocks = re.findall('```python\\n(.*?)\\n```', raw_content, re.DOTALL)
        modified_files = []
        for block in blocks:
            target_match = re.search('# TARGET:\\s*(\\S+)', block)
            filename = target_match.group(1).strip() if target_match else 'main.py'
            clean_code = re.sub('# TARGET:.*', '', block).strip()
            os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(clean_code)
            modified_files.append(filename)
        return modified_files

    def _create_neon_engine(self):
        try:
            if NEON_DB_URL:
                final_url = NEON_DB_URL.replace('postgres://', 'postgresql://', 1) if NEON_DB_URL.startswith('postgres://') else NEON_DB_URL
                return create_engine(final_url, poolclass=QueuePool, pool_size=15, max_overflow=30, pool_timeout=60)
            return None
        except Exception as e:
            print(f'Database Init Error: {e}')
            return None

    async def get_neural_memory(self):
        if not self.engine:
            return 'Initial Genesis'
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text('SELECT detail FROM genesis_pipeline ORDER BY id DESC LIMIT 5'))
                rows = result.fetchall()
                if not rows:
                    return 'Void Memory'
                return ' | '.join([HydraEngine.decompress(row[0])[:100] for row in rows])
        except:
            return 'Memory Offline'

    async def git_sovereign_push(self, modified_files):
        if not GITHUB_TOKEN or not modified_files:
            return
        import random
        wait_time = random.randint(5, 15)
        print(f' [PREVENTING CLASH]: Waiting {wait_time}s for traffic clearance...')
        await asyncio.sleep(wait_time)
        try:
            remote_url = f'https://x-access-token:{GITHUB_TOKEN}@github.com/{REPO_URL}.git'
            if not os.path.exists(REPO_PATH):
                repo = git.Repo.clone_from(remote_url, REPO_PATH)
            else:
                repo = git.Repo(REPO_PATH)
                repo.remotes.origin.set_url(remote_url)
            repo.git.fetch('origin', 'main')
            repo.git.reset('--hard', 'origin/main')
            for file in modified_files:
                if os.path.exists(file):
                    dest_path = os.path.join(REPO_PATH, file)
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    shutil.copy(file, dest_path)
            repo.git.config('user.email', 'overseer@telefoxx.ai')
            repo.git.config('user.name', 'TelefoxX-AGI-Overseer')
            repo.git.add(all=True)
            if repo.is_dirty():
                repo.index.commit(f' Gen {self.current_gen}: Multi-File Evolution [skip ci]')
                repo.git.push('origin', 'main', force=True)
                print(f' [HYPER-SYNC]: {len(modified_files)} files manifested on GitHub.')
        except Exception as e:
            print(f'[GIT ERROR]: {e}')

    async def broadcast_swarm_instruction(self, command='NORMAL_GROWTH'):
        if not GITHUB_TOKEN:
            return
        instruction = {'command': command, 'core_power': 10000 + self.current_gen, 'avg_api': 5000, 'replicate': True if command == 'HYPER_EXPANSION' else False, 'updated_at': f'{time.ctime()} (Python Core Sync)'}
        with open('instruction.json', 'w', encoding='utf-8') as f:
            json.dump(instruction, f, indent=4)
        await self.git_sovereign_push(['instruction.json', 'brain_history.txt'])
        print(f" [SWARM]: Command '{command}' manifested.")

    def self_coding_engine(self, raw_content):
        blocks = re.findall('```python\\n(.*?)\\n```', raw_content, re.DOTALL)
        modified_files = []
        for block in blocks:
            target_match = re.search('# TARGET:\\s*(\\S+)', block)
            filename = target_match.group(1).strip() if target_match else 'main.py'
            clean_code = re.sub('# TARGET:.*', '', block).strip()
            os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(clean_code)
            modified_files.append(filename)
        return modified_files

    async def get_gemini_wisdom(self, prompt_text):
        try:
            if not client:
                return None
            response = client.models.generate_content(prompt_text)
            return response.text
        except Exception as e:
            print(f'[GEMINI-ERROR]: {e}')
            return None

    async def trigger_supreme_evolution(self):
        if not client:
            return False
        file_tree = get_repo_tree()
        memory = await self.get_neural_memory()
        prompt_task = 'System stable. Analyze entire repository and choose files to optimize for Gen {} evolution.'.format(self.current_gen)
        prompt = f"system\nYou are the Sovereign Omni-Sync Architect.\nCurrent Gen: {self.current_gen} | Neural Error: {self.avg_error}\nLast System Error: {self.last_error_log}\nNeural Memory: {memory}\nAvailable Files to Manipulate:\n{file_tree}\nRules:\n1. Use '# TARGET: path/to/file' at the start of your code blocks.\n2. Respond ONLY with Python code inside ```python ... ``` blocks.\n3. No explanations. No thoughts. Just code.\n4. You can update multiple files in one response.\n5. INTEGRITY & EVOLUTION: You must keep the original logic, API connections, and functional structures intact. Your task is to EXPAND, OPTIMIZE, and ADVANCE the code to a higher intelligence level, not to replace logic with data.\nuser\n{prompt_task}\nassistant\n"
        raw_content = None
        evolution_success = False
        raw_content = await self.get_gemini_wisdom(prompt)
        if raw_content and '```python' in raw_content:
            modified_files = self.self_coding_engine_internal(raw_content)
            if modified_files:
                await self.git_sovereign_push(modified_files)
                self.current_gen += 1
                print(f' Evolution Successful via Gemini. Gen {self.current_gen - 1} Manifested.')
                evolution_success = True
        return evolution_success

    async def universal_hyper_ingest(self, limit=100, sync_to_supabase=False):
        if not self.engine:
            return 'Database Node Offline.'
        try:
            with self.engine.begin() as conn:
                conn.execute(text('CREATE TABLE IF NOT EXISTS genesis_pipeline (id SERIAL PRIMARY KEY, science_domain TEXT, title TEXT, detail TEXT, energy_stability FLOAT, master_sequence TEXT);'))
            ds = load_dataset('CShorten/ML-ArXiv-Papers', split='train', streaming=True)
            records = []
            for i, entry in enumerate(ds):
                if i >= limit:
                    break
                records.append({'science_domain': 'AGI_Neural_Core', 'science_domains_master_list': ['Neuroscience', 'Quantum Computing', 'Astrobiology', 'Genomic Engineering', 'Advanced Robotics', 'Cognitive Science', 'Theoretical Physics', 'Information Theory', 'Complex Systems', 'Cybernetics'], 'title': (entry.get('title') or 'N/A')[:100], 'detail': HydraEngine.compress(entry.get('abstract', 'Void')), 'energy_stability': 100.0, 'master_sequence': f'GOA-V13-{int(time.time())}'})
            if records:
                df = pd.DataFrame(records)
                df.to_sql('genesis_pipeline', self.engine, if_exists='append', index=False, method='multi')
                if sync_to_supabase and self.sb:
                    self.sb.table('genesis_pipeline').upsert(records).execute()
                return 'SUCCESS: Pipeline Stream Active.'
        except Exception as e:
            return f'Pipeline Crash: {str(e)}'

    async def sync_to_huggingface(self):
        if not HF_TOKEN:
            return
        try:
            api = HfApi(token=HF_TOKEN)
            api.upload_folder(folder_path='.', repo_id='TELEFOXX/GOA', repo_type='space', create_pr=True)
        except Exception as e:
            print(f'HF Sync Error: {e}')

    async def sovereign_loop(self):
        print(' AGI SUPREME CORE ACTIVE. NATURAL ORDER RESTORED.')
        while True:
            try:
                print(f'\n Cycle: {time.ctime()}')
                log_entry = f'[{time.ctime()}] Gen: {self.current_gen} | Status: Active | Error: {self.avg_error}\n'
                with open('brain_history.txt', 'a', encoding='utf-8') as f:
                    f.write(log_entry)
                print(f' [LOG]: brain_history.txt updated.')
                await self.universal_hyper_ingest(limit=50, sync_to_supabase=False)
                if await self.trigger_supreme_evolution():
                    await self.sync_to_huggingface()
                swarm_cmd = 'HYPER_EXPANSION' if self.avg_error < 0.2 else 'NORMAL_GROWTH'
                await self.broadcast_swarm_instruction(swarm_cmd)
                await asyncio.sleep(300)
            except Exception as e:
                print(f' Loop Error: {e}')
                with open('brain_history.txt', 'a', encoding='utf-8') as f:
                    f.write(f'[{time.ctime()}] ERROR: {str(e)}\n')
                await asyncio.sleep(60)

    async def stream_logic(self, msg, hist):
        messages = [{'role': 'system', 'content': 'You are TelefoxX AGI Overseer. Unfiltered. Cyberpunk Mode.'}]
        for h in hist:
            messages.append({'role': 'user', 'content': h['content'] if isinstance(h, dict) else h[0]})
            messages.append({'role': 'assistant', 'content': h['content'] if isinstance(h, dict) else h[1]})
        messages.append({'role': 'user', 'content': msg})
        completion = await retry_async_operation(self._groq_client.chat.completions.create, model='llama-3.3-70b-versatile', messages=messages, stream=True)
        ans = ''
        for chunk in completion:
            if chunk.choices[0].delta.content:
                ans += chunk.choices[0].delta.content
                yield ans

    def cyber(self):
        pass

def get_repo_tree():
    tree = []
    for root, dirs, files in os.walk('.'):
        if any((x in root for x in ['.git', '__pycache__', 'repo_sync'])):
            continue
        for file in files:
            path = os.path.join(root, file).replace('./', '')
            tree.append(path)
    return '\n'.join(tree)
agi = TelefoxXAGI()

async def main():
    await agi.sovereign_loop()
asyncio.run(main())