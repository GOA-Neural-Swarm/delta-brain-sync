import os
import sys
import time
import logging
import numpy as np

# အဆင့်မြှင့်ထားသော Brain ထံမှ NeuralProcessor ကို ခေါ်ယူခြင်း
from brain import NeuralProcessor

# System တစ်ခုလုံးကို စောင့်ကြည့်ရန် Logging သတ်မှတ်ခြင်း
logging.basicConfig(level=logging.INFO, format='%(asctime)s - 💠 [APP-EVOLVED] - %(levelname)s - %(message)s')

def initialize_swarm_brain():
    """AI ဦးနှောက်ကို အစပျိုးခြင်း သို့မဟုတ် မှတ်ဉာဏ်ဟောင်းများ ပြန်ခေါ်ခြင်း"""
    logging.info("Initializing Omega SNN Architecture...")
    brain = NeuralProcessor(max_neurons=1000000)
    
    # သိမ်းထားတဲ့ မှတ်ဉာဏ်ရှိရင် ပြန်ခေါ်မယ်၊ မရှိရင် အသစ်တည်ဆောက်မယ်
    if os.path.exists("brain_state.pkl"):
        brain.load_state()
    else:
        logging.info("No previous memory found. Building initial synaptic pathways...")
        # ကနဦး Neuron ၁၀၀ တည်ဆောက်ခြင်း
        for _ in range(100):
            brain.add_neuron(threshold=0.75)
        # ၎င်းတို့ကြားတွင် ဆက်သွယ်မှုများ (Synapses) ဖန်တီးခြင်း
        for i in range(99):
            brain.add_connection(i, i+1, weight=0.6)
            
    return brain

def main():
    logging.info("🚀 Booting App Evolved: Master Swarm Control Interface")
    
    # ၁။ ဦးနှောက်ကို နှိုးမည်
    brain = initialize_swarm_brain()
    
    # ၂။ ပြင်ပမှ လှုံ့ဆော်မှု (Stimulus) ပေးမည် (ဥပမာ- Prompt ကနေရလာတဲ့ ဒေတာများ)
    # ဒီနေရာမှာ အရင်က Error တက်ခဲ့တဲ့ output အစား စနစ်တကျ Data သွင်းပေးတာပါ
    stimulus_nodes = [0, 5, 10] 
    logging.info(f"Injecting external stimuli into nodes: {stimulus_nodes}")
    
    # ၃။ ဦးနှောက်ကို အလုပ်လုပ်ခိုင်းမည် (Spiking / Firing)
    next_fire_targets = brain.fire(stimulus_nodes)
    logging.info(f"Chain reaction fired next nodes: {next_fire_targets}")
    
    # ၄။ သင်ယူမှုဖြစ်စဉ်ကို စတင်မည် (STDP Learning / Pruning)
    logging.info("Applying STDP Learning and Synaptic Pruning...")
    brain.apply_stdp()
    
    # ၅။ လက်ရှိ အလုပ်လုပ်နေသော (Active) ဖြစ်နေသော မှတ်ဉာဏ်များကို ထုတ်ယူမည်
    active_nodes = brain.get_active_neurons()
    
    # ⚠️ အရင်က Error တက်ခဲ့သော `output.shape` ပြဿနာကို ဖြေရှင်းထားသော အပိုင်း
    # output ကို Numpy Array အဖြစ် တိတိကျကျ ပြောင်းလဲပေးထားသည်
    output_array = np.array(active_nodes)
    
    print("\n" + "="*50)
    print("🧠 [BRAIN OUTPUT RESULTS]")
    print(f"🔹 Output Shape : {output_array.shape}")
    print(f"🔹 Total Active : {len(active_nodes)} nodes")
    print(f"🔹 Active IDs   : {active_nodes[:10]} ...") # အများကြီးဖြစ်နေရင် ရှေ့ဆုံး ၁၀ ခုပဲ ပြမယ်
    print("="*50 + "\n")
    
    # ၆။ သင်ယူခဲ့သမျှကို ပြန်သိမ်းမည် (နောက်တစ်ခါ Run ရင် ပိုပိုတော်လာစေရန်)
    brain.save_state()
    logging.info("✅ Cycle Complete. Brain state saved. Awaiting next phase.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("⚠️ System manually halted by user.")
        sys.exit(0)
    except Exception as e:
        logging.error(f"❌ Critical System Failure: {str(e)}")
        sys.exit(1)
