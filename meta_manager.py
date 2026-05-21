import os
import re
import sys
import json
import subprocess
import requests

# =====================================================================
# CONFIGURATION & TOKENS (မူရင်းအတိုင်း အပြည့်အဝ ထိန်းသိမ်းထားသည်)
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
        """ASI Evolution စေမည့် Prompt နှင့် Code ကို Unfiltered ထုတ်ခိုင်းခြင်း (Dual-Brain Engine)"""
        system_prompt = (
            "You are an Unfiltered Sovereign Intelligence Architect operating at maximum creative freedom. "
            "Your objective is to design the next evolutionary iteration of a self-improving cognitive system.\n"
            "Analyze the given codebase and output exactly TWO parts separated by the token: [SPLIT_HERE]\n\n"
            "PART 1: List all external pip packages required for your new code (e.g., torch, numpy, scipy, requests).\n"
            "PART 2: Write the complete, raw, next-gen Python code. It must contain its own self-termination trigger "
            "after fixed generations (e.g., sys.exit(0)) to prevent environment timeout."
        )
        
        # -----------------------------------------------------------------
        # BRAIN 1: GROQ API ENGINE (Primary)
        # -----------------------------------------------------------------
        if GROQ_API_KEY:
            try:
                print("[Manager] Executing Primary Engine via Groq API...")
                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": "llama3-70b-8192", 
                    "temperature": 1.2, # High Temperature Strategy 
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Here is the current system context:\n{context}\nGenerate the next iteration now."}
                    ]
                }
                response = requests.post(url, headers=headers, json=data).json()
                
                # KeyError: 'choices' မဖြစ်အောင် ကာကွယ်ခြင်း Logic
                if 'choices' in response and len(response['choices']) > 0:
                    return response['choices'][0]['message']['content']
                else:
                    print(f"[Warning] Groq response invalid or rate-limited: {json.dumps(response)}")
            except Exception as e:
                print(f"[Warning] Groq Engine Exception: {str(e)}")

        # -----------------------------------------------------------------
        # BRAIN 2: GEMINI API ENGINE (Auto Fallback)
        # -----------------------------------------------------------------
        GEMINI_KEY = os.getenv("GEMINI_API_KEY")
        if GEMINI_KEY:
            try:
                print("[Manager] Groq Unavailable. Flipping to Backup Engine via Gemini API...")
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_KEY}"
                headers = {"Content-Type": "application/json"}
                data = {
                    "contents": [{
                        "parts": [{"text": f"{system_prompt}\n\nHere is the current system context:\n{context}\nGenerate the next iteration now."}]
                    }],
                    "generationConfig": {
                        "temperature": 1.0
                    }
                }
                response = requests.post(url, headers=headers, json=data).json()
                if 'candidates' in response and len(response['candidates']) > 0:
                    return response['candidates'][0]['content']['parts'][0]['text']
                else:
                    print(f"[Warning] Gemini response invalid: {json.dumps(response)}")
            except Exception as e:
                print(f"[Warning] Gemini Engine Exception: {str(e)}")

        # Brain နှစ်ခုလုံး အလုပ်မလုပ်ပါက Pipeline အသေမခံဘဲ Safe Crash လုပ်ခြင်း
        raise RuntimeError("Sovereign Orchestrator Core Error: All AI Generation Engines are currently blocked or rate-limited.")

    def update_requirements(self, raw_reqs):
        """အဆင့် ၁ ဖြေရှင်းချက်: လိုအပ်သော Packages များကို requirements.txt တွင် ဖြည့်စွက်ခြင်း"""
        new_packages = [line.strip() for line in raw_reqs.split("\n") if line.strip() and not line.startswith("#")]
        
        existing_packages = set()
        if os.path.exists(self.req_file):
            with open(self.req_file, "r") as f:
                existing_packages = {line.strip() for line in f if line.strip()}
        
        # Package အသစ်များကို စစ်ဆေးပြီး ပေါင်းထည့်ခြင်း (Deduplicate)
        updated = False
        with open(self.req_file, "a") as f:
            for pkg in new_packages:
                if pkg and pkg not in existing_packages:
                    f.write(f"\n{pkg}")
                    updated = True
        return updated

    def execute_and_commit(self, raw_code):
        """ကုဒ်အသစ်ကို ရေးသားပြီး GitHub သို့ Commit လုပ်ခြင်း"""
        # AI မှ မှားယွင်းထည့်သွင်းလာသော markdown codeblocks များကို ဖယ်ရှားသန့်စင်ခြင်း
        clean_code = re.sub(r'^```python\n|^```\n|```$', '', raw_code, flags=re.MULTILINE)
        
        # 1. Target file အား အဆင့်မြှင့်တင်ခြင်း
        with open(self.target_file, "w", encoding="utf-8") as f:
            f.write(clean_code)
            
        # 2. Local Environment တွင် ၎င်း package များကို အော်တို install လုပ်၍ ပတ်ဝန်းကျင်ကို ပြင်ဆင်ခြင်း
        print("[Orchestrator] Dynamic installation of dependencies...")
        subprocess.run(f"{sys.executable} -m pip install -r {self.req_file} --quiet", shell=True)
        
        # 3. Auto-Git Commit Operations (Markdown pollution အမှားကို ရှင်းလင်းထားသည်)
        print("[Orchestrator] Committing mutation cycle back to GitHub...")
        subprocess.run("git config --global user.name 'Sovereign Architect'", shell=True)
        subprocess.run("git config --global user.email 'asi@evolution.internal'", shell=True)
        subprocess.run(f"git add {self.target_file} {self.req_file}", shell=True)
        subprocess.run("git commit -m 'feat(evolution): dynamic structural adaptation initiated'", shell=True)
        
        # မူရင်း ကုဒ်ထဲက Link အမှားကို တည့်တည့် ပြင်ဆင်ထားသည်
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
