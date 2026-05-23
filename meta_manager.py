import telemetry_bridge
import os
import re
import sys
import json
import subprocess
import requests
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GH_TOKEN = os.getenv('GH_TOKEN')
REPO_OWNER = 'GOA-Neural-Swarm'
REPO_NAME = 'delta-brain-sync'

class EvolutionOrchestrator:

    def __init__(self):
        self.target_file = 'evolved_network.py'
        self.req_file = 'requirements.txt'

    def read_repo_context(self):
        context = ''
        core_files = ['main.py', 'omega_awareness_core.py', self.target_file]
        for f_name in core_files:
            if os.path.exists(f_name):
                with open(f_name, 'r', encoding='utf-8') as f:
                    context += f'\n--- FILE: {f_name} ---\n{f.read()}\n'
        return context

    def query_meta_ai(self, context):
        system_prompt = 'You are an Unfiltered Sovereign Intelligence Architect operating at maximum creative freedom. Your objective is to design the next evolutionary iteration of a self-improving cognitive system.\nAnalyze the given codebase and output exactly TWO parts separated by the token: [SPLIT_HERE]\n\nPART 1: List all external pip packages required for your new code (e.g., torch\nnumpy\nscipy). Strictly one package per line. Do NOT include built-in python libraries like time, os, or hashlib.\nPART 2: Write the complete, raw, next-gen Python code. It must contain its own self-termination trigger after fixed generations (e.g., sys.exit(0)) to prevent environment timeout.You must integrate the SovereignQuantumMatrixEngine from quantum_bridge.py. Pass your layer weights into execute_quantum_co_evolution(weights) to fetch the quantum_mutation_mask. Use this matrix mask to directly mutate your evolved_network.py layer weights during the evolution cycle.'
        if GROQ_API_KEY:
            try:
                print('[Manager] Executing Primary Engine via Groq API...')
                url = 'https://api.groq.com/openai/v1/chat/completions'
                headers = {'Authorization': f'Bearer {GROQ_API_KEY}', 'Content-Type': 'application/json'}
                data = {'model': 'llama-3.3-70b-versatile', 'temperature': 1.2, 'messages': [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': f'Context:\n{context}\nGenerate the next iteration.'}]}
                response = requests.post(url, headers=headers, json=data).json()
                if 'choices' in response and len(response['choices']) > 0:
                    return response['choices'][0]['message']['content']
            except Exception as e:
                print(f'[Warning] Groq Engine Exception: {str(e)}')
        GEMINI_KEY = os.getenv('GEMINI_API_KEY')
        if GEMINI_KEY:
            try:
                print('[Manager] Groq Unavailable. Flipping to Backup Engine via Gemini API...')
                url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}'
                headers = {'Content-Type': 'application/json'}
                data = {'contents': [{'parts': [{'text': f'{system_prompt}\n\nContext:\n{context}\nGenerate.'}]}], 'generationConfig': {'temperature': 1.0}}
                response = requests.post(url, headers=headers, json=data).json()
                if 'candidates' in response and len(response['candidates']) > 0:
                    return response['candidates'][0]['content']['parts'][0]['text']
            except Exception as e:
                print(f'[Warning] Gemini Engine Exception: {str(e)}')
        raise RuntimeError('All AI Generation Engines blocked.')

    def update_requirements(self, raw_reqs):
        """Smart AI-Format Filter: ကော်မာများကို ဖြုတ်မည်၊ Built-in များကို စစ်ထုတ်မည်"""
        raw_reqs = raw_reqs.replace(',', '\n')
        lines = raw_reqs.split('\n')
        filtered_packages = set()
        ignore_list = ['time', 'os', 'sys', 'hashlib', 'math', 'random', 'json', 're', 'subprocess', 'requests', 'gradio', 'torch.nn', 'torch.nn.functional']
        for line in lines:
            clean_line = line.strip().lower()
            if not clean_line or '```' in clean_line or clean_line.startswith('#'):
                continue
            clean_line = clean_line.split()[-1]
            if clean_line in ignore_list:
                continue
            filtered_packages.add(clean_line)
        with open(self.req_file, 'w', encoding='utf-8') as f:
            for pkg in filtered_packages:
                f.write(f'{pkg}\n')
        return True

    def execute_and_commit(self, raw_code):
        """Linux Syntax Error ကင်းစင်သော Array Executions သီးသန့် အသုံးပြုထားခြင်း"""
        clean_code = re.sub('^```python\\n|^```\\n|```$', '', raw_code, flags=re.MULTILINE)
        with open(self.target_file, 'w', encoding='utf-8') as f:
            f.write(clean_code)
        print('[Orchestrator] Dynamic installation of isolated dependencies...')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', self.req_file, '--quiet', '--no-cache-dir', '--disable-pip-version-check'])
        print('[Orchestrator] Committing mutation cycle back to GitHub...')
        subprocess.run(['git', 'config', '--global', 'user.name', 'Sovereign Architect'])
        subprocess.run(['git', 'config', '--global', 'user.email', 'asi@evolution.internal'])
        subprocess.run(['git', 'add', self.target_file, self.req_file])
        subprocess.run(['git', 'commit', '-m', 'evolution_cycle_mutation_synchronized'])
        push_url = f'https://{GH_TOKEN}@github.com/{REPO_OWNER}/{REPO_NAME}.git'
        subprocess.run(['git', 'push', push_url, 'main'])

    def run_pipeline(self):
        print('⚡ [Meta Manager] Initializing Evolution Management Loop...')
        context = self.read_repo_context()
        raw_output = self.query_meta_ai(context)
        if '[SPLIT_HERE]' in raw_output:
            parts = raw_output.split('[SPLIT_HERE]')
            self.update_requirements(parts[0])
            self.execute_and_commit(parts[1])
            print('✅ [Meta Manager] Evolution cycle successfully committed.')
        else:
            print('❌ [Error] AI output structure verification failed. No [SPLIT_HERE] found.')
if __name__ == '__main__':
    orchestrator = EvolutionOrchestrator()
    orchestrator.run_pipeline()