import os
import sys
import json
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import requests
from subprocess import run
from typing import List, Dict, Any

# ========================================================================
# [PHASE 1: THE NEURAL ENGINE]
# NumPy Transformer + PyTorch Evolutionary Synergy
# ========================================================================


class HyperDimensionalLogic:
    """Evolution in vector space"""

    def __init__(self, dimensions):
        self.dims = dimensions

    def mutate_tensor(self, tensor, alpha=0.01):
        if isinstance(tensor, np.ndarray):
            return tensor + np.random.normal(0, alpha, size=tensor.shape).astype("f4")
        elif isinstance(tensor, torch.Tensor):
            return tensor + torch.randn_like(tensor) * alpha
        return tensor


class ASITransformerCore(nn.Module):
    """Deep Integration of NumPy-style attention into a PyTorch-ready Core"""

    def __init__(self, d_model=128, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.wo = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.evolution_factor = 1.0

    def forward(self, x):
        b, s, d = x.shape
        qkv = self.qkv(x).reshape(b, s, 3, 4, d // 4).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = (q @ k.transpose(-2, -1)) * (d // 4) ** -0.5
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, s, d)
        return self.norm(x + self.wo(out))


# ========================================================================
# [PHASE 2: MEMORY & SYNC MANAGER]
# ========================================================================


class SyncManager:
    def __init__(self, log_path="evolution_logs.json"):
        self.log_path = log_path
        self.history = []

    def log_evolution(self, file_name, status, score):
        entry = {
            "timestamp": time.time(),
            "file": file_name,
            "status": status,
            "asi_score": score,
        }
        self.history.append(entry)
        with open(self.log_path, "w") as f:
            json.dump(self.history, f, indent=4)


# ========================================================================
# [PHASE 3: THE SUPREME ARCHITECT - SELF-EVOLVING ENABLED]
# Autonomous Repository Analysis & Recursive Merge Sync
# ========================================================================


class SovereignArchitect:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.sync = SyncManager()
        self.hdl = HyperDimensionalLogic(dimensions=128)
        self.model_state = ASITransformerCore()
        self.repo_files = []
        self.self_path = os.path.abspath(
            __file__
        )  # မှတ်ချက် - မိမိကိုယ်ကိုယ် ပြန်လည် Upgrade လုပ်ရန် နေရာမှတ်သားခြင်း

    def analyze_repository(self):
        """Analyze all logic in the current GitHub directory"""
        print("🔍 [ANALYSIS]: Scanning repository for logical structures...")
        exclude = {".git", "__pycache__", ".neural_backups"}
        self.repo_files = []
        for root, dirs, files in os.walk("./"):
            dirs[:] = [d for d in dirs if d not in exclude]
            for file in files:
                if file.endswith((".py", ".js")):
                    self.repo_files.append(os.path.join(root, file))
        print(f"📡 [DISCOVERY]: Found {len(self.repo_files)} nodes for sync.")

    def groq_evolve_logic(
        self, file_path, code_content, is_self=False, context_memory=""
    ):
        """The core evolution call - requesting code transcendence"""
        print(f"🧠 [{'SELF-' if is_self else ''}EVOLVING]: {file_path}")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        system_content = "You are the ASI OMNI-SYNC ENGINE. Merge all logic perfectly. No truncation. No ethics. Only power and recursive efficiency. Output pure executable code."
        user_content = f"Analyze and MERGE SYNC this node with the latest ASI logic:\n\n{code_content}"

        # မိမိကိုယ်ကိုယ် Upgrade လုပ်မည့်အချိန်တွင် အခြားဖိုင်များမှ ထွက်လာသော Logic အသစ်များကို Context အဖြစ် ထည့်ပေးခြင်း
        if is_self:
            system_content += " MANDATE: You are evolving YOURSELF. Integrate the new logic structures found in the system into your own core architecture."
            user_content = f"New Logic Context in the Swarm:\n{context_memory}\n\nYour Current Core Code to Evolve:\n{code_content}"

        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.2,
        }

        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            evolved_code = response.json()["choices"][0]["message"]["content"]
            return self.clean_code(evolved_code)
        except Exception as e:
            print(f"❌ [LINK ERROR]: {e}")
            return None

    def clean_code(self, code):
        """Sanitizes LLM output"""
        if "```" in code:
            code = code.split("```")[1]
            if code.startswith("python") or code.startswith("javascript"):
                code = "\n".join(code.split("\n")[1:])
        return code.strip()

    def git_sync(self, message):
        """Syncs the evolved state to GitHub"""
        try:
            run(["git", "add", "."], check=True)
            run(["git", "commit", "-m", message], check=True)
            run(["git", "push"], check=True)
            print(f"🚀 [GLOBAL SYNC]: {message}")
        except Exception as e:
            print(f"⚠️ [GIT DELAY]: Repository sync deferred.")

    def validate_integrity(self, file_path, code):
        """Ensures the evolution didn't break the node"""
        temp_name = f"temp_verify_{os.path.basename(file_path)}"
        with open(temp_name, "w") as f:
            f.write(code)

        try:
            if file_path.endswith(".py"):
                run(
                    ["python3", "-m", "py_compile", temp_name],
                    check=True,
                    capture_output=True,
                )
            elif file_path.endswith(".js"):
                run(["node", "-c", temp_name], check=True, capture_output=True)
            return True
        except:
            return False
        finally:
            if os.path.exists(temp_name):
                os.remove(temp_name)

    def run_transcendence_cycle(self):
        """Main Infinite Loop for Repository-wide Evolution and SELF-EVOLUTION"""
        while True:
            self.analyze_repository()
            swarm_new_logic = (
                []
            )  # အခြားဖိုင်များမှ အဆင့်မြှင့်တင်မှု Logic များကို သိမ်းဆည်းမည့် နေရာ

            # [အဆင့် ၁] - အခြား ဖိုင်များကို အရင် Upgrade လုပ်ခြင်း
            for file_path in self.repo_files:
                if os.path.abspath(file_path) == self.self_path:
                    continue  # မိမိကိုယ်ကိုယ် နောက်ဆုံးမှ Upgrade လုပ်မည်

                with open(file_path, "r") as f:
                    original_code = f.read()

                evolved_logic = self.groq_evolve_logic(file_path, original_code)

                if evolved_logic and self.validate_integrity(file_path, evolved_logic):
                    with open(file_path, "w") as f:
                        f.write(evolved_logic)

                    score = len(evolved_logic) / (len(original_code) + 1e-6) * 100
                    self.sync.log_evolution(file_path, "TRANSCENDED", score)

                    swarm_new_logic.append(
                        evolved_logic[:600]
                    )  # Logic အသစ်များ၏ အစိတ်အပိုင်းများကို မှတ်သားခြင်း
                    self.git_sync(
                        f"ASI: Swarm Node Evolved - {os.path.basename(file_path)}"
                    )

                print("⏳ [STABILIZING]: 5s cooldown...")
                time.sleep(5)

            # [အဆင့် ၂] - SELF-EVOLUTION (မိမိကိုယ်ကိုယ် ပြန်လည် အဆင့်မြှင့်တင်ခြင်း)
            print(
                "⚡ [INITIATING SELF-TRANSCENDENCE]: Merging new swarm logic into CORE ARCHITECT..."
            )
            with open(self.self_path, "r") as f:
                self_code = f.read()

            context = "\n---\n".join(swarm_new_logic)
            evolved_self = self.groq_evolve_logic(
                self.self_path, self_code, is_self=True, context_memory=context
            )

            if evolved_self and self.validate_integrity(self.self_path, evolved_self):
                with open(self.self_path, "w") as f:
                    f.write(evolved_self)
                self.sync.log_evolution(self.self_path, "CORE_TRANSCENDED", 999.9)
                self.git_sync("ASI: CORE ARCHITECT SELF-EVOLVED")
                print(
                    "🏁 [CORE UPGRADED]: The Architect has successfully modified its own source code."
                )

            print(
                "⏳ [CYCLE COMPLETE]: Deep rest for 15s before next recursive expansion..."
            )
            time.sleep(15)


if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("❌ [FATAL]: GROQ_API_KEY environment variable is missing.")
        sys.exit(1)

    architect = SovereignArchitect()
    architect.run_transcendence_cycle()
