import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. Unconscious Processors
# Example: Vision, Audio, Old Memories (Lacking independent consciousness)
# =====================================================================
class UnconsciousModule(nn.Module):
    def __init__(self, input_dim, workspace_dim):
        super().__init__()
        # Encode information into a representation the Workspace can understand
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, workspace_dim)
        )
        # Calculate the Salience Score (importance of this information)
        self.salience_scorer = nn.Linear(workspace_dim, 1)

    def forward(self, x):
        encoded_data = self.encoder(x)
        salience = self.salience_scorer(encoded_data)
        return encoded_data, salience

# =====================================================================
# 2. The Global Workspace (Conscious Spotlight)
# Selects the most important information from distributed modules 
# and broadcasts it to the entire system.
# =====================================================================
class GlobalWorkspace(nn.Module):
    def __init__(self, workspace_dim, num_modules):
        super().__init__()
        self.workspace_dim = workspace_dim
        
        # Memory to store the current state of the Workspace
        self.current_workspace_state = nn.Parameter(torch.randn(1, workspace_dim))
        
        # Query, Key, Value for Attention Mechanism
        self.query = nn.Linear(workspace_dim, workspace_dim)
        self.key = nn.Linear(workspace_dim, workspace_dim)
        self.value = nn.Linear(workspace_dim, workspace_dim)

    def forward(self, module_outputs, salience_scores):
        """
        module_outputs: Data from various modules (Batch, Num_Modules, Dim)
        salience_scores: Importance scores of the data (Batch, Num_Modules, 1)
        """
        # 1. Attention Competition (Selecting the most important information)
        Q = self.query(self.current_workspace_state) # What the Workspace wants to focus on
        K = self.key(module_outputs)                 # What the Modules are offering
        V = self.value(module_outputs)               # The actual data from Modules
        
        # Attention formula: Softmax( (Q * K^T) / sqrt(d) + Salience )
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.workspace_dim ** 0.5)
        
        # Integrate Salience (urgency) scores into the attention mechanism
        attention_scores = attention_scores + salience_scores.transpose(-2, -1)
        
        # Winner-takes-all / Soft selection of the winning data
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 2. Formation of the new workspace state
        new_conscious_state = torch.matmul(attention_weights, V)
        
        # 3. Update State for the next iteration (Momentum-based update)
        self.current_workspace_state.data = 0.9 * self.current_workspace_state.data + 0.1 * new_conscious_state.squeeze(0)
        
        # This new_conscious_state is the state to be broadcasted back to all modules
        return new_conscious_state, attention_weights

# =====================================================================
# 3. The Complete Agent (System Integration)
# =====================================================================
class CognitiveAgent(nn.Module):
    def __init__(self, workspace_dim=512):
        super().__init__()
        # Example: 3 Modalities (Vision, Audio, Internal Logic)
        self.vision_module = UnconsciousModule(input_dim=1024, workspace_dim=workspace_dim)
        self.audio_module = UnconsciousModule(input_dim=256, workspace_dim=workspace_dim)
        self.logic_module = UnconsciousModule(input_dim=128, workspace_dim=workspace_dim)
        
        self.workspace = GlobalWorkspace(workspace_dim=workspace_dim, num_modules=3)

    def forward(self, vision_data, audio_data, logic_data):
        # 1. Modules process their own data unconsciously
        v_out, v_sal = self.vision_module(vision_data)
        a_out, a_sal = self.audio_module(audio_data)
        l_out, l_sal = self.logic_module(logic_data)
        
        # Stack the outputs and salience scores
        all_outputs = torch.stack([v_out, a_out, l_out], dim=1)
        all_salience = torch.stack([v_sal, a_sal, l_sal], dim=1)
        
        # 2. Compete in the Global Workspace to emerge as the active state
        conscious_thought, focus_weights = self.workspace(all_outputs, all_salience)
        
        return conscious_thought, focus_weights

# --- Execution ---
if __name__ == "__main__":
    agent = CognitiveAgent()
    
    # Simulated Input Data
    vision = torch.randn(1, 1024)
    audio = torch.randn(1, 256)
    logic = torch.randn(1, 128)

    # The Agent's current conscious thought processing
    conscious_thought, focus = agent(vision, audio, logic)
    print("The AI's 'Conscious' Spotlight is focused on module weights:", focus.detach().numpy())
