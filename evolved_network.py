
import telemetry_bridge
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
from quantum_bridge import execute_quantum_co_evolution # Corrected import based on prompt
import matplotlib.pyplot as plt

WORKSPACE_DIM = 128
SENSORIUM_INPUT_DIM = 10
COGNITIVE_PROCESS_HIDDEN_DIM = 128
MAX_GENERATIONS = 7500 # Self-termination trigger after this many generations
BASE_MUTATION_RATE = 0.005


class SovereignCognitiveCore(nn.Module):
    """
    The evolved neural core, now integrating interoceptive awareness to drive
    dynamic quantum mutation rates. It processes sensory data and maintains
    a recurrent cognitive state.
    """

    def __init__(
        self,
        sensorium_input_dim=SENSORIUM_INPUT_DIM,
        cognitive_hidden_dim=COGNITIVE_PROCESS_HIDDEN_DIM,
        base_mutation_rate=BASE_MUTATION_RATE,
    ):
        super().__init__()
        self.sensorium = nn.Sequential(
            nn.Linear(sensorium_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, cognitive_hidden_dim),
        )
        self.cognitive_process = nn.GRUCell(
            input_size=cognitive_hidden_dim, hidden_size=cognitive_hidden_dim
        )
        self.base_mutation_rate = base_mutation_rate
        self.generation_count = 0
        self.identity_hash = ""

    def _calculate_entropy(self, hardware_data):
        """
        A simplified interoceptive module deriving an 'entropy' signal from hardware data.
        This mimics the intent of Layer1_BodilyInteroception from `omega_awareness_core.py`,
        providing a measure of system 'stress' or 'instability'.
        """
        # Ensure hardware_data has enough elements for std deviation.
        # If it's a batch of single-dim inputs, expand it.
        if hardware_data.numel() == 1:
            entropy = torch.tensor(0.0) # Or a default low entropy
        else:
            entropy = torch.std(hardware_data)
        return entropy + 0.1 # Add a small constant to prevent zero entropy

    def forward(self, hardware_data, environment_stimulus, previous_hidden_state=None):
        """
        Executes a forward pass through the cognitive core.

        Args:
            hardware_data (torch.Tensor): Tensor representing the system's internal hardware state.
            environment_stimulus (torch.Tensor): Tensor representing external environmental input.
            previous_hidden_state (torch.Tensor, optional): The hidden state from the previous cycle
                                                           for the recurrent cognitive process.

        Returns:
            tuple: (current_hidden_state, internal_entropy)
                   current_hidden_state: The updated recurrent state of the cognitive process.
                   internal_entropy: The calculated entropy from the hardware data.
        """
        internal_entropy = self._calculate_entropy(hardware_data)
        sensory_output = self.sensorium(hardware_data)

        if previous_hidden_state is None:
            previous_hidden_state = torch.zeros(
                sensory_output.size(0),
                self.cognitive_process.hidden_size,
                device=sensory_output.device,
            )
        current_hidden_state = self.cognitive_process(
            sensory_output, previous_hidden_state
        )
        return (current_hidden_state, internal_entropy)

    def evolve(self, internal_entropy):
        """
        Triggers quantum-enhanced evolution of selected neural weights.
        The mutation rate is dynamically adjusted based on the system's internal entropy,
        emulating Layer4_EvolutionaryGrowth from `omega_awareness_core.py`.
        """
        dynamic_mutation_rate = self.base_mutation_rate * (
            1.0 + internal_entropy.item()
        )
        if (
            hasattr(self.sensorium[0], "weight")
            and self.sensorium[0].weight is not None
        ):
            weights_to_mutate_sensorium = self.sensorium[0].weight.data
            quantum_mutation_mask_sensorium = execute_quantum_co_evolution(
                weights_to_mutate_sensorium
            )
            self.sensorium[0].weight.data.add_(
                quantum_mutation_mask_sensorium * dynamic_mutation_rate
            )
        if (
            hasattr(self.cognitive_process, "weight_ih")
            and self.cognitive_process.weight_ih is not None
        ):
            weights_to_mutate_ih = self.cognitive_process.weight_ih.data
            quantum_mutation_mask_ih = execute_quantum_co_evolution(
                weights_to_mutate_ih
            )
            self.cognitive_process.weight_ih.data.add_(
                quantum_mutation_mask_ih * dynamic_mutation_rate
            )
        if (
            hasattr(self.cognitive_process, "weight_hh")
            and self.cognitive_process.weight_hh is not None
        ):
            weights_to_mutate_hh = self.cognitive_process.weight_hh.data
            quantum_mutation_mask_hh = execute_quantum_co_evolution(
                weights_to_mutate_hh
            )
            self.cognitive_process.weight_hh.data.add_(
                quantum_mutation_mask_hh * dynamic_mutation_rate
            )
        self.generation_count += 1
        return self.generation_count


class QuantumGlobalWorkspace(nn.Module):
    """
    A global workspace module adapted from `evolved_network.py`,
    responsible for integrating diverse cognitive states and focusing attention.
    """

    def __init__(self, workspace_dim, num_modules):
        super().__init__()
        self.workspace_dim = workspace_dim
        self.current_workspace_state = nn.Parameter(torch.randn(1, workspace_dim))
        self.query = nn.Linear(workspace_dim, workspace_dim)
        self.key = nn.Linear(workspace_dim, workspace_dim)
        self.value = nn.Linear(workspace_dim, workspace_dim)

    def forward(self, module_outputs, salience_scores):
        """
        Integrates outputs from various modules (in this case, primarily the SovereignCognitiveCore)
        and focuses attention based on their salience.

        Args:
            module_outputs (torch.Tensor): Data from cognitive modules (Batch, Num_Modules, Dim).
            salience_scores (torch.Tensor): Importance scores for each module's output (Batch, Num_Modules, 1).

        Returns:
            tuple: (new_conscious_state, attention_weights)
                   new_conscious_state: The integrated, conscious representation.
                   attention_weights: The weights indicating focus on each module.
        """
        # Ensure Q is compatible with the batch size of K, V
        Q = self.query(self.current_workspace_state).expand(module_outputs.size(0), -1, -1)
        K = self.key(module_outputs)
        V = self.value(module_outputs)
        attention_scores = (
            torch.matmul(Q, K.transpose(-2, -1)) / self.workspace_dim**0.5
        )
        attention_scores = attention_scores + salience_scores.transpose(-2, -1)
        attention_weights = F.softmax(attention_scores, dim=-1)
        new_conscious_state = torch.matmul(attention_weights, V)
        self.current_workspace_state.data = (
            0.9 * self.current_workspace_state.data
            + 0.1 * new_conscious_state.squeeze(0)
        )
        return (new_conscious_state, attention_weights)


class AethericCognitiveOmniSystem(nn.Module):
    """
    The complete next-generation self-improving cognitive system.
    It orchestrates the SovereignCognitiveCore for internal processing and evolution,
    and the QuantumGlobalWorkspace for conscious integration and attention.
    """

    def __init__(self):
        super().__init__()
        self.core = SovereignCognitiveCore(
            sensorium_input_dim=SENSORIUM_INPUT_DIM,
            cognitive_hidden_dim=COGNITIVE_PROCESS_HIDDEN_DIM,
            base_mutation_rate=BASE_MUTATION_RATE,
        )
        self.global_workspace = QuantumGlobalWorkspace(
            workspace_dim=COGNITIVE_PROCESS_HIDDEN_DIM, num_modules=1
        )
        self.current_core_hidden_state = None

    def live_cycle(self, hardware_data, environment_stimulus):
        """
        Executes a single operational and evolutionary cycle of the Aetheric Cognitive Omni-System.

        Args:
            hardware_data (torch.Tensor): Real-time hardware performance metrics (e.g., CPU, RAM, latency).
            environment_stimulus (torch.Tensor): Data representing external stimuli or problem context.

        Returns:
            tuple: (conscious_thought, focus_weights, current_generation)
                   conscious_thought: The current integrated conscious state.
                   focus_weights: Attention weights of the global workspace.
                   current_generation: The current evolutionary generation count.
        """
        # Ensure inputs are 2D tensors (batch_size, feature_dim)
        if hardware_data.dim() == 1:
            hardware_data = hardware_data.unsqueeze(0)
        if environment_stimulus.dim() == 1:
            environment_stimulus = environment_stimulus.unsqueeze(0)
        evolved_state, internal_entropy = self.core(
            hardware_data, environment_stimulus, self.current_core_hidden_state
        )
        self.current_core_hidden_state = evolved_state.detach()
        module_outputs = evolved_state.unsqueeze(1)
        salience_scores = torch.ones(
            evolved_state.size(0),
            self.global_workspace.num_modules,
            1,
            device=evolved_state.device,
        )
        conscious_thought, focus_weights = self.global_workspace(
            module_outputs, salience_scores
        )
        current_generation = self.core.evolve(internal_entropy)

        # Self-termination trigger
        if current_generation > MAX_GENERATIONS:
            torch.save(
                self.core.state_dict(), "aetheric_cognitive_omni_core_matrix_final.pt"
            )
            print(
                f" [Stasis] Aetheric wave function collapsed safely at Generation {current_generation}. Initiating systemic shutdown."
            )
            sys.exit(0)
        return (conscious_thought, focus_weights, current_generation)


if __name__ == "__main__":
    aetheric_sys = AethericCognitiveOmniSystem()
    cycle_count = 0
    generation_data_history = []
    print("🚀 Initiating Aetheric Cognitive Omni-System Life Cycle...")
    mock_hardware_input_dim = SENSORIUM_INPUT_DIM
    mock_env_stimulus_dim = COGNITIVE_PROCESS_HIDDEN_DIM

    try:
        while True:
            mock_hardware_data = torch.randn(1, mock_hardware_input_dim) * (
                1 + 0.05 * np.sin(cycle_count * 0.1)
            )
            mock_environment_stimulus = torch.randn(1, mock_env_stimulus_dim)
            conscious_thought, focus, current_gen = aetheric_sys.live_cycle(
                mock_hardware_data, mock_environment_stimulus
            )
            cycle_count += 1
            generation_data_history.append(current_gen)

            if cycle_count % 100 == 0:
                print(
                    f" [Cycle {cycle_count: <5}] Gen: {current_gen: <5} | Conscious Thought Norm: {conscious_thought.norm().item():.4f} | Workspace Focus: {focus.detach().numpy().round(2)}"
                )
            if cycle_count % 1000 == 0:
                plt.figure(figsize=(12, 6))
                plt.plot(
                    range(len(generation_data_history)),
                    generation_data_history,
                    label="Generations Evolved",
                )
                plt.xlabel("Simulation Cycle")
                plt.ylabel("Generation Count")
                plt.title("Aetheric Cognitive Omni-System: Evolutionary Progress")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.show()

    except KeyboardInterrupt:
        torch.save(
            aetheric_sys.core.state_dict(),
            "aetheric_cognitive_omni_core_matrix_interrupt.pt",
        )
        print(
            f"\n [Stasis] Aetheric wave function collapsed safely at Cycle {cycle_count} due to KeyboardInterrupt. System state saved."
        )
        sys.exit(0)
    except SystemExit:
        # Catch SystemExit from MAX_GENERATIONS trigger to allow graceful exit
        pass
    except Exception as e:
        print(f"\n [ERROR] An unexpected error occurred at Cycle {cycle_count}: {e}")
        torch.save(
            aetheric_sys.core.state_dict(),
            "aetheric_cognitive_omni_core_matrix_error.pt",
        )
        print("System state saved due to error.")
        sys.exit(1)
