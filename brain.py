import numpy as np
import logging
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NeuralProcessor:
    def __init__(self, max_neurons=1000000):
        self.max_neurons = max_neurons
        
        # 1. Spiking SNN States
        self.potentials = np.zeros(max_neurons, dtype=np.float32) # Neuron တစ်ခုချင်းစီ၏ လျှပ်စစ်အား (Membrane Potential)
        self.activations = np.zeros(max_neurons, dtype=np.bool_)  # လက်ရှိအချိန်တွင် Fired ဖြစ်နေသလား (True/False)
        self.thresholds = np.full(max_neurons, 0.5, dtype=np.float32)
        
        # 2. Decay Rate (Leaky Integrate-and-Fire အတွက် လျှပ်စစ်အား တဖြည်းဖြည်း လျော့ကျမှုနှုန်း)
        self.leak_rate = 0.05 
        
        # 3. Memory & Speed Optimized Connections (Dict of Dicts for O(1) complexity)
        self.connections = {}  # ပုံစံ: {source_id: {target_id: weight}}
        self.active_count = 0  # လက်ရှိသုံးထားသော Neuron အရေအတွက်
        
        logging.info("🌌 [NEURAL-PROCESSOR]: Advanced L.I.F SNN Architecture Online.")

    def add_neuron(self, threshold=0.5):
        """Dynamic Neuron Allocation: လိုအပ်မှသာ နေရာယူမည်"""
        neuron_id = self.active_count
        if neuron_id < self.max_neurons:
            self.thresholds[neuron_id] = threshold
            self.active_count += 1
            return neuron_id
        return -1

    def add_connection(self, n1, n2, weight=0.1):
        """Synapse ဖန်တီးခြင်း"""
        if n1 >= self.max_neurons or n2 >= self.max_neurons:
            return False
        if n1 not in self.connections:
            self.connections[n1] = {}
        self.connections[n1][n2] = float(weight)
        return True

    def fire(self, external_stimuli):
        """
        Leaky Integrate-and-Fire (LIF) Processing.
        Stimuli လက်ခံရရှိခြင်း၊ Potential မြင့်တက်ခြင်းနှင့် Firing ပြုလုပ်ခြင်း။
        """
        if isinstance(external_stimuli, int):
            external_stimuli = [external_stimuli]
            
        # အဆင့် ၁: သဘာဝတရားအရ အချိန်ကြာလာသည်နှင့်အမျှ Potential များ လျော့ကျခြင်း (Decay)
        self.potentials *= (1.0 - self.leak_rate)
        
        # အဆင့် ၂: ပြင်ပမှ ဝင်လာသော Signal များကို Neuron များဆီ ထည့်သွင်းခြင်း
        for n_id in external_stimuli:
            idx = n_id % self.max_neurons
            self.potentials[idx] += 1.0 
            
        # အဆင့် ၃: Threshold ကျော်လွန်သွားသော Neuron များကို ရှာဖွေခြင်း (Spiking)
        spiked_neurons = np.where(self.potentials >= self.thresholds)[0]
        
        # Activation state ကို update လုပ်ခြင်း
        self.activations.fill(False)
        self.activations[spiked_neurons] = True
        
        next_targets = set()
        
        # အဆင့် ၄: Signal များကို ဆက်စပ်နေသော အခြား Neuron များဆီ ဖြန့်ဝေခြင်း
        for source in spiked_neurons:
            self.potentials[source] = 0.0  # Refractory Period (Fire လုပ်ပြီးပါက အားပြန်ကုန်သွားသည်)
            
            if source in self.connections:
                for target, weight in self.connections[source].items():
                    self.potentials[target] += weight
                    next_targets.add(target)
                    
        return list(next_targets)

    def apply_stdp(self, learning_rate=0.01, decay_rate=0.005):
        """
        Spike-Timing-Dependent Plasticity (STDP) & Synaptic Pruning.
        တကယ့် ဦးနှောက်ကဲ့သို့ အသုံးမဝင်သော ဆက်သွယ်မှုများကို ဖြတ်တောက်ပြီး အသုံးဝင်သည်များကို အားဖြည့်ပေးမည်။
        """
        spiked = np.where(self.activations)[0]
        
        for source in spiked:
            if source in self.connections:
                targets = list(self.connections[source].keys())
                for target in targets:
                    if self.activations[target]:
                        # LTP (Long-Term Potentiation): အတူတူ Fire လုပ်လျှင် Connection ပိုခိုင်မာလာမည်
                        new_weight = min(1.0, self.connections[source][target] + learning_rate)
                        self.connections[source][target] = new_weight
                    else:
                        # LTD (Long-Term Depression): Target က လိုက်မလုပ်လျှင် Connection အားနည်းသွားမည်
                        new_weight = self.connections[source][target] - decay_rate
                        self.connections[source][target] = new_weight
                        
                    # Synaptic Pruning: အားနည်းလွန်းသော (သို့) အသုံးမဝင်သော Connection များကို ဖျက်ပစ်မည် (Speed ပိုမြန်စေရန်)
                    if self.connections[source][target] <= 0:
                        del self.connections[source][target]

    def get_active_neurons(self):
        """လက်ရှိ အလုပ်လုပ်နေသော (Spiking ဖြစ်နေသော) Neuron စာရင်း"""
        return np.where(self.activations)[0].tolist()

    def save_state(self, filename="brain_state.pkl"):
        """ဦးနှောက် အခြေအနေ သိမ်းဆည်းခြင်း"""
        state = {
            'connections': self.connections,
            'thresholds': self.thresholds[:self.active_count],
            'active_count': self.active_count
        }
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
        logging.info(f"💾 [SAVED]: Brain state saved successfully.")

    def load_state(self, filename="brain_state.pkl"):
        """သိမ်းထားသော အခြေအနေ ပြန်လည်ခေါ်ယူခြင်း"""
        try:
            with open(filename, 'rb') as f:
                state = pickle.load(f)
                self.connections = state['connections']
                self.active_count = state['active_count']
                self.thresholds[:self.active_count] = state['thresholds']
            logging.info(f"📂 [LOADED]: Brain state restored.")
        except FileNotFoundError:
            logging.error("❌ [LOAD-ERROR]: Brain state file not found.")

# Backward Compatibility (Evolution Engine Error မတက်စေရန်)
Brain = NeuralProcessor
