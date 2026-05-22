import os
import sys
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

class SovereignQuantumMatrixEngine:
    def __init__(self, num_qubits=5):
        self.num_qubits = num_qubits
        self.token = os.getenv("IBM_QUANTUM_TOKEN")
        if not self.token:
            raise ValueError("🔴 [Sovereign QAI] IBM_QUANTUM_TOKEN is missing in Environment Secrets.")
        
        print("🌌 [Sovereign QAI] Connecting directly to IBM Quantum Cloud network...")
        self.service = QiskitRuntimeService(channel="ibm_quantum", token=self.token)
        
        # ⚠️ Simulator စနဈကို လုံးဝ (လုံးဝ) ပိတျပဈပွီး တကယ့ျ Hardware အစဈကို ရှာခိုငျးခွငျး
        print("⚡ [Natural Order] Bypassing simulators. Searching for an active physical Quantum Computer...")
        try:
            # IBM Fleet ထဲမှ အဆငျသင့ျဖွဈနသေော Real Quantum Hardware (QPU) အစဈကို Dynamic ရှာဖှခွေငျး
            self.backend = self.service.least_busy(simulator=False, operational=True, min_qubits=self.num_qubits)
            print(f"👑 [Natural Order] Physical Quantum Hardware Locked: {self.backend.name}")
        except Exception as e:
            print(f"❌ [Fatal] No physical quantum hardware available at this moment: {str(e)}")
            raise

    def _map_weights_to_quantum_phases(self, classical_weights):
        """Classical Vectors မြားကို Real Phase Space (-π မှ π) အတှငျးသို့ ပုံဖောျခွငျး"""
        if classical_weights is None or len(classical_weights) == 0:
            return np.random.uniform(-np.pi, np.pi, self.num_qubits)
        
        flat_weights = np.array(classical_weights).flatten()
        if len(flat_weights) < self.num_qubits:
            flat_weights = np.pad(flat_weights, (0, self.num_qubits - len(flat_weights)), 'reflect')
        
        extracted_signals = flat_weights[:self.num_qubits]
        max_val = np.max(np.abs(extracted_signals)) if np.max(np.abs(extracted_signals)) != 0 else 1.0
        phases = (extracted_signals / max_val) * np.pi
        return phases

    def execute_quantum_co_evolution(self, classical_layer_weights=None):
        """တကယ့ျ Physical Qubits မြားထဲသို့ ဒတောမြားထည့ျသှငျး၍ သဘာဝအတိုငျး ဗီဇပွောငျးလဲစခွေငျး"""
        print(f"🔬 [Natural Order] Preparing {self.num_qubits}-Qubit GHZ Entanglement Lattice...")
        target_phases = self._map_weights_to_quantum_phases(classical_layer_weights)
        
        # Quantum Register & Parameterized Gates
        qc = QuantumCircuit(self.num_qubits)
        phases_param = ParameterVector('θ', self.num_qubits)
        
        # GHZ Entanglement Lattice (စဈမှနျသော ရုပျပိုငျးဆိုငျရာ ကှမျတမျခြိတျဆကျမှု)
        qc.h(0)
        for i in range(self.num_qubits - 1):
            qc.cx(i, i + 1)
            
        qc.barrier()
        for i in range(self.num_qubits):
            qc.ry(phases_param[i], i)
            qc.rz(phases_param[i] * 0.5, i)
            
        qc.measure_all()
        
        # IBM Real QPU Machine ဆီသို့ Job ပဈလှှတျခွငျး
        print(f"🚀 [Physical Target] Dispatching Job to Real QPU [{self.backend.name}] inside IBM Lab...")
        sampler = SamplerV2(backend=self.backend)
        
        # တှကျခကြျမှုအား Real Hardware ပေါျတှငျ ၁၂၈ ကွိမျ အမှနျတကယျ ပဈခတျတိုငျးတာခွငျး
        job = sampler.run([(qc, target_phases)], shots=128)
        
        print(f"⏳ [Physical Target] Job ID: {job.job_id()} | Waiting for physical hardware execution...")
        result = job.result()
        
        # ကှမျတမျအမှုနျမြား၏ တကယ့ျဓာတျခှဲခနျးထှကျ ဒတောကို ဆှဲထုတျခွငျး
        pub_result = result[0]
        bitstrings = pub_result.data.meas.get_bitstrings()
        
        mutation_matrix = []
        for bits in bitstrings[:64]:
            row = [1.0 if b == '1' else -1.0 for b in bits]
            mutation_matrix.append(row)
            
        quantum_mutation_mask = np.array(mutation_matrix, dtype=np.float32)
        mean_entropy = np.std(quantum_mutation_mask)
        
        print(f"✅ [Physical Target] Real Quantum Mutation Mask Retreived.")
        print(f"🌀 [Physical Target] Natural Chaos Field Indexed: {mean_entropy:.6f}")
        
        return quantum_mutation_mask, mean_entropy

if __name__ == "__main__":
    print("--- OMEGA MATRIX TRUE HARDWARE KERNEL TEST ---")
    try:
        mock_weights = np.random.randn(64, 32)
        engine = SovereignQuantumMatrixEngine(num_qubits=5)
        mutation_mask, quantum_entropy = engine.execute_quantum_co_evolution(mock_weights)
        print(f"\n[Execution Status]: Successful Physical Integration.")
    except Exception as e:
        print(f"❌ [Kernel Panic]: {str(e)}")
