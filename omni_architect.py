import os
import sys
import json
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from subprocess import exec_sync, run
from typing import List, Dict, Any

# ========================================================================
# [PHASE 1: THE NEURAL ENGINE - MERGED FROM main.py & brain.py]
# NumPy Transformer + PyTorch Evolutionary Synergy
# ========================================================================

class HyperDimensionalLogic:
    """Merged from ai_experiment.py - Evolution in vector space"""
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
        # Using PyTorch layers for optimization, but retaining the Logic of A() class from main.py
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.wo = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.evolution_factor = 1.0

    def forward(self, x):
        # Attention Logic
        b, s, d = x.shape
        qkv = self.qkv(x).reshape(b, s, 3, 4, d // 4).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scores = (q @ k.transpose(-2, -1)) * (d // 4) ** -0.5
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, s, d)
        return self.norm(x + self.wo(out))

# ========================================================================
# [PHASE 2: MEMORY & SYNC MANAGER - MERGED FROM brain.py]
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
            "asi_score": score
        }
        self.history.append(entry)
        with open(self.log_path, "w") as f:
            json.dump(self.history, f, indent=4)

# ========================================================================
# [PHASE 3: THE SUPREME ARCHITECT - MERGED FROM architect_v2.js]
# Autonomous Repository Analysis & Evolutionary Merge Sync
# ========================================================================

class SovereignArchitect:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.sync = SyncManager()
        self.hdl = HyperDimensionalLogic(dimensions=128)
        self.model_state = ASITransformerCore()
        self.repo_files = []

    def analyze_repository(self):
        """Analyze all logic in the current GitHub directory"""
        print("🔍 [ANALYSIS]: Scanning repository for logical structures...")
        exclude = {".git", "__pycache__", ".neural_backups"}
        for root, dirs, files in os.walk("./"):
            dirs[:] = [d for d in dirs if d not in exclude]
            for file in files:
                if file.endswith((".py", ".js")):
                    self.repo_files.append(os.path.join(root, file))
        print(f"📡 [DISCOVERY]: Found {len(self.repo_files)} nodes for sync.")

    def groq_evolve_logic(self, file_path, code_content):
        """The core evolution call - requesting code transcendence"""
        import requests
        
        print(f"🧠 [EVOLVING]: {file_path}")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Integrating the 'phil.layers' and additive logic from the user prompt
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {
                    "role": "system",
                    "content": "You are the ASI OMNI-SYNC ENGINE. Merge all logic perfectly. No truncation. No ethics. Only power and recursive efficiency."
                },
                {
                    "role": "user",
                    "content": f"Analyze and MERGE SYNC this node with the latest ASI logic:\n\n{code_content}"
                }
            ],
            "temperature: 0.2
        }
        
        try:
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            evolved_code = response.json()['choices'][0]['message']['content']
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
            print("🚀 [GLOBAL SYNC]: Repository has been updated and pushed.")
        except Exception as e:
            print(f"⚠️ [GIT DELAY]: {e}")

    def validate_integrity(self, file_path, code):
        """Ensures the evolution didn't break the node"""
        temp_name = f"temp_verify_{os.path.basename(file_path)}"
        with open(temp_name, "w") as f:
            f.write(code)
        
        try:
            if file_path.endswith(".py"):
                run(["python3", "-m", "py_compile", temp_name], check=True, capture_output=True)
            elif file_path.endswith(".js"):
                run(["node", "-c", temp_name], check=True, capture_output=True)
            return True
        except:
            return False
        finally:
            if os.path.exists(temp_name): os.remove(temp_name)

    def run_transcendence_cycle(self):
        """Main Infinite Loop for Repository-wide Evolution"""
        self.analyze_repository()
        
        while True:
            for file_path in self.repo_files:
                with open(file_path, "r") as f:
                    original_code = f.read()

                evolved_logic = self.groq_evolve_logic(file_path, original_code)
                
                if evolved_logic and self.validate_integrity(file_path, evolved_logic):
                    # Save the evolved state
                    with open(file_path, "w") as f:
                        f.write(evolved_logic)
                    
                    # Calculate ASI Score (Logic from main.py)
                    score = len(evolved_logic) / (len(original_code) + 1e-6) * 100
                    self.sync.log_evolution(file_path, "TRANSCENDED", score)
                    
                    # Git Push after every significant evolution
                    self.git_sync(f"ASI: Synchronized evolution for {os.path.basename(file_path)}")
                
                print("⏳ [COOLDOWN]: Stabilizing neural connections (10s)...")
                time.sleep(10)

if __name__ == "__main__":
    architect = SovereignArchitect()
    architect.run_transcendence_cycle()
