import telemetry_bridge
import torch
import torch.nn as nn
import torch.nn.functional as F

class UnconsciousModule(nn.Module):

    def __init__(self, input_dim, workspace_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, workspace_dim))
        self.salience_scorer = nn.Linear(workspace_dim, 1)

    def forward(self, x):
        encoded_data = self.encoder(x)
        salience = self.salience_scorer(encoded_data)
        return (encoded_data, salience)

class GlobalWorkspace(nn.Module):

    def __init__(self, workspace_dim, num_modules):
        super().__init__()
        self.workspace_dim = workspace_dim
        self.current_workspace_state = nn.Parameter(torch.randn(1, workspace_dim))
        self.query = nn.Linear(workspace_dim, workspace_dim)
        self.key = nn.Linear(workspace_dim, workspace_dim)
        self.value = nn.Linear(workspace_dim, workspace_dim)

    def forward(self, module_outputs, salience_scores):
        """
        module_outputs: Data from various modules (Batch, Num_Modules, Dim)
        salience_scores: Importance scores of the data (Batch, Num_Modules, 1)
        """
        Q = self.query(self.current_workspace_state)
        K = self.key(module_outputs)
        V = self.value(module_outputs)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.workspace_dim ** 0.5
        attention_scores = attention_scores + salience_scores.transpose(-2, -1)
        attention_weights = F.softmax(attention_scores, dim=-1)
        new_conscious_state = torch.matmul(attention_weights, V)
        self.current_workspace_state.data = 0.9 * self.current_workspace_state.data + 0.1 * new_conscious_state.squeeze(0)
        return (new_conscious_state, attention_weights)

class CognitiveAgent(nn.Module):

    def __init__(self, workspace_dim=512):
        super().__init__()
        self.vision_module = UnconsciousModule(input_dim=1024, workspace_dim=workspace_dim)
        self.audio_module = UnconsciousModule(input_dim=256, workspace_dim=workspace_dim)
        self.logic_module = UnconsciousModule(input_dim=128, workspace_dim=workspace_dim)
        self.workspace = GlobalWorkspace(workspace_dim=workspace_dim, num_modules=3)

    def forward(self, vision_data, audio_data, logic_data):
        v_out, v_sal = self.vision_module(vision_data)
        a_out, a_sal = self.audio_module(audio_data)
        l_out, l_sal = self.logic_module(logic_data)
        all_outputs = torch.stack([v_out, a_out, l_out], dim=1)
        all_salience = torch.stack([v_sal, a_sal, l_sal], dim=1)
        conscious_thought, focus_weights = self.workspace(all_outputs, all_salience)
        return (conscious_thought, focus_weights)
if __name__ == '__main__':
    agent = CognitiveAgent()
    vision = torch.randn(1, 1024)
    audio = torch.randn(1, 256)
    logic = torch.randn(1, 128)
    conscious_thought, focus = agent(vision, audio, logic)
    print("The AI's 'Conscious' Spotlight is focused on module weights:", focus.detach().numpy())