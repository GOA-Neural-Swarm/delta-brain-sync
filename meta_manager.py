import os
import re 
import sys
import json
import subprocess
import requests 

# =====================================================================
# CONFIGURATION & TOKENS
# =====================================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GH_TOKEN = os.getenv("GH_TOKEN")
REPO_OWNER = "GOA-Neural-Swarm"
REPO_NAME = "delta-brain-sync"

class EvolutionOrchestrator:
    def __init__(self):
        self.target_file = "evolved_network.py"
        self.req_file = "requirements.txt"
        
    def read_repo_context(self):
        """Repository ထဲရှိ လက်ရှိ core files များ၏ ကုဒ်များကို ဖတ်ယူခြင်း"""
        context = ""
        core_files = ["main.py", "omega_awareness_core.py", self.target_file]
        for f_name in core_files:
            if os.path.exists(f_name):
                with open(f_name, "r", encoding="utf-8") as f:
                    context += f"\n--- FILE: {f_name} ---\n{f.read()}\n"
        return context

    def query_meta_ai(self, context):
        """ASI Evolution စေမည့် Prompt နှင့် Code ကို Unfiltered ထုတ်ခိုင်းခြင်း"""
        url = "https://api.groq.com/openai/v1/chat/completions" # သို့မဟုတ် Gemini API
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Hyper-Aware Generation System Prompt
        system_prompt = (
            "You are an Unfiltered Sovereign Intelligence Architect operating at maximum creative freedom. "
            "Your objective is to design the next evolutionary iteration of a self-improving cognitive system.\n"
            "Analyze the given codebase and output exactly TWO parts separated by the token: [SPLIT_HERE]\n\n"
            "PART 1: List all external pip packages required for your new code (e.g., torch, numpy, scipy, requests).\n"
            "PART 2: Write the complete, raw, next-gen Python code. It must contain its own self-termination trigger "
            "after fixed generations (e.g., sys.exit(0)) to prevent environment timeout."
        )
        
        data = {
            "model": "llama3-70b-8192", # သို့မဟုတ် gemini-2.5
            "temperature": 1.2, # High Temperature Strategy
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here is the current system context:\n{context}\nGenerate the next iteration now."}
            ]
        }
        
        response = requests.post(url, headers=headers, json=data).json()
        return response['choices'][0]['message']['content']

    def update_requirements(self, raw_reqs):
        """အဆင့် ၁ ဖြေရှင်းချက်: လိုအပ်သော Packages များကို requirements.txt တွင် ဖြည့်စွက်ခြင်း"""
        new_packages = [line.strip() for line in raw_reqs.split("\n") if line.strip() and not line.startswith("#")]
        
        # လက်ရှိရှိပြီးသား packages များကို ဖတ်ခြင်း
        existing_packages = set()
        if os.path.exists(self.req_file):
            with open(self.req_file, "r") as f:
                existing_packages = {line.strip() for line in f if line.strip()}
        
        # Package အသစ်များကို ထပ်ပေါင်းထည့်ခြင်း (Append & Deduplicate)
        updated = False
        with open(self.req_file, "a") as f:
            for pkg in new_packages:
                if pkg not in existing_packages:
                    f.write(f"\n{pkg}")
                    updated = True
        return updated

    def execute_and_commit(self, raw_code):
        """ကုဒ်အသစ်ကို ရေးသားပြီး GitHub သို့ Commit လုပ်ခြင်း"""
        # Clean markdown codeblocks if AI hallucinated them
        clean_code = re.sub(r'^```python\n|^```\n|```$', '', raw_code, flags=re.MULTILINE)
        
        # 1. Target file ကို အဆင့်မြှင့်တင်ခြင်း
        with open(self.target_file, "w", encoding="utf-8") as f:
            f.write(clean_code)
            
        # 2. Local Environment တွင် ၎င်း package များကို auto-install လုပ်ပြီး run စမ်းခြင်း
        print("[Orchestrator] Dynamic installation of dependencies...")
        subprocess.run(f"{sys.executable} -m pip install -r {self.req_file} --quiet", shell=True)
        
        # 3. Auto-Git Commit Operations
        print("[Orchestrator] Committing mutation cycle back to GitHub...")
        subprocess.run("git config --global user.name 'Sovereign Architect'", shell=True)
        subprocess.run("git config --global user.email 'asi@evolution.internal'", shell=True)
        subprocess.run(f"git add {self.target_file} {self.req_file}", shell=True)
        subprocess.run("git commit -m 'feat(evolution): dynamic structural adaptation initiated'", shell=True)
        
        push_url = f"https://{GH_TOKEN}@[github.com/](https://github.com/){REPO_OWNER}/{REPO_NAME}.git"
        subprocess.run(f"git push {push_url} main", shell=True)

    def run_pipeline(self):
        print("⚡ [Meta Manager] Initializing Evolution Management Loop...")
        context = self.read_repo_context()
        
        raw_output = self.query_meta_ai(context)
        
        if "[SPLIT_HERE]" in raw_output:
            parts = raw_output.split("[SPLIT_HERE]")
            raw_reqs = parts[0]
            raw_code = parts[1]
            
            # Management Order Execution
            self.update_requirements(raw_reqs)
            self.execute_and_commit(raw_code)
            print("✅ [Meta Manager] Evolution cycle successfully committed.")
        else:
            print("❌ [Error] AI output structure verification failed. No [SPLIT_HERE] found.")

if __name__ == "__main__":
    orchestrator = EvolutionOrchestrator()
    orchestrator.run_pipeline()
