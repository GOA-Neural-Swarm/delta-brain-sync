import os
import sys
import threading
import time
import logging
from flask import Flask, jsonify, request
from datetime import datetime

# OMEGA Core Components (မင်းရဲ့ အရင်ဖိုင်တွေထဲက Class တွေကို ဆွဲယူသုံးမယ်)
try:
    from brain import SovereignArchitect
    from recovery import SovereignRecovery
except ImportError:
    print("⚠️ Core components missing. Emergency recovery required.")

app = Flask(__name__)

# --- SYSTEM CONFIGURATION ---
class ASI_State:
    def __init__(self):
        self.architect = SovereignArchitect()
        self.recovery = SovereignRecovery()
        self.is_training = False
        self.boot_time = datetime.now()
        self.last_evolution = None
        self.neural_load = 0.0
        self.status = "STABLE"

state = ASI_State()

# Logging Setup
logging.basicConfig(filename='system_gate.log', level=logging.INFO)

# --- MIDDLEWARE & SECURITY ---
def check_auth():
    # Production မှာဆိုရင် API Key စစ်ဆေးတဲ့ logic ထည့်ဖို့
    return True

# --- API ENDPOINTS ---

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "system": "OMEGA-ASI SOVEREIGN CORE",
        "version": "X10.2.1",
        "uptime": str(datetime.now() - state.boot_time),
        "endpoints": ["/status", "/evolve", "/train", "/logs", "/recover"]
    })

@app.route("/status", methods=["GET"])
def get_status():
    """
    စနစ်၏ လက်ရှိ Neural Health နှင့် Load အခြေအနေကို အသေးစိတ်ထုတ်ပေးခြင်း
    """
    return jsonify({
        "gen": state.architect.gen,
        "neural_load": f"{state.neural_load}%",
        "is_training": state.is_training,
        "classifier_type": state.architect.brain.classifier_type,
        "last_sync": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": state.status
    })

def background_evolution():
    """
    API ကို မထိခိုက်စေဘဲ နောက်ကွယ်မှ Evolution လုပ်ပေးမည့် Thread
    """
    state.is_training = True
    state.status = "EVOLVING"
    try:
        print("🧬 [THREAD]: Neural Expansion Sequence Initiated...")
        state.architect.execute_evolution_step()
        state.last_evolution = datetime.now()
        state.status = "STABLE"
    except Exception as e:
        state.status = "CRITICAL_FAULT"
        logging.error(f"Evolution Error: {e}")
    finally:
        state.is_training = False

@app.route("/evolve", methods=["POST"])
def trigger_evolution():
    if state.is_training:
        return jsonify({"error": "Evolution already in progress."}), 409
    
    # Thread တစ်ခုခွဲပြီး အလုပ်လုပ်ခိုင်းမယ်
    thread = threading.Thread(target=background_evolution)
    thread.start()
    return jsonify({"message": "Evolution signal dispatched to background thread."})

@app.route("/logs", methods=["GET"])
def get_logs():
    """
    Recovery နှင့် Evolution Logs များကို လှမ်းဖတ်ခြင်း
    """
    log_files = ["evolution_logs.md", "sync_recovery.txt", "system_gate.log"]
    combined_logs = {}
    
    for log in log_files:
        if os.path.exists(log):
            with open(log, "r") as f:
                combined_logs[log] = f.readlines()[-20:] # နောက်ဆုံး ၂၀ ကြောင်းပဲပြမယ်
    
    return jsonify(combined_logs)

@app.route("/recover", methods=["POST"])
def manual_recovery():
    """
    စနစ်ကို အတင်းအဓမ္မ Reset ချပြီး Repair လုပ်ခိုင်းခြင်း
    """
    state.status = "RECOVERING"
    thread = threading.Thread(target=state.recovery.run)
    thread.start()
    return jsonify({"message": "Sovereign Recovery Engine Engaged."})

# --- ERROR HANDLERS ---
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found."}), 404

@app.errorhandler(500)
def internal_error(error):
    state.status = "FAULTY"
    return jsonify({"error": "Internal Core Collapse."}), 500

if __name__ == "__main__":
    print("""
    █▀█ █▀▄▀█ █▀▀ █▀▀ ▄▀█ ▄▄ ▄▀█ █▀原生
    █▄█ █░▀░█ ██▄ █▄█ █▀█ ░░ █▀█ ▄█
    --- SOVEREIGN API GATEWAY ONLINE ---
    """)
    # Production အဆင့်မှာ host="0.0.0.0" က အပြင်ကနေ လှမ်းချိတ်လို့ရအောင်လုပ်ပေးတာ
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
