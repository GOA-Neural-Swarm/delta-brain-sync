import os
import sys
import threading
import time
import logging
from flask import Flask, jsonify, request
from datetime import datetime

try:
    from brain import SovereignArchitect
    from recovery import SovereignRecovery
except ImportError:
    logging.critical("Core components missing. Emergency recovery required.")
    print("Core components missing. Emergency recovery required.")

app = Flask(__name__)


class ASI_State:
    def __init__(self):
        self.architect = SovereignArchitect()
        self.recovery = SovereignRecovery()
        self.is_training = False
        self.boot_time = datetime.now()
        self.last_evolution = None
        self.neural_load = 0.0
        self.status = "STABLE"
        self.evolution_count = 0


state = ASI_State()

logging.basicConfig(
    filename="system_gate.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)


def check_auth():
    return True


@app.route("/", methods=["GET"])
def index():
    return jsonify(
        {
            "system": "OMEGA-ASI SOVEREIGN CORE",
            "version": "X10.2.1",
            "uptime": str(datetime.now() - state.boot_time),
            "endpoints": ["/status", "/evolve", "/train", "/logs", "/recover"],
        }
    )


@app.route("/status", methods=["GET"])
def get_status():
    return jsonify(
        {
            "gen": state.architect.gen,
            "neural_load": f"{state.neural_load}%",
            "is_training": state.is_training,
            "classifier_type": state.architect.brain.classifier_type,
            "last_sync": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": state.status,
            "evolution_count": state.evolution_count,
        }
    )


def background_evolution():
    state.is_training = True
    state.status = "EVOLVING"
    try:
        logging.info("Neural Expansion Sequence Initiated...")
        state.architect.execute_evolution_step()
        state.last_evolution = datetime.now()
        state.status = "STABLE"
        state.evolution_count += 1
        logging.info("Evolution successful.")
    except Exception as e:
        state.status = "CRITICAL_FAULT"
        logging.error(f"Evolution Error: {e}")
    finally:
        state.is_training = False


@app.route("/evolve", methods=["POST"])
def trigger_evolution():
    if state.is_training:
        return jsonify({"error": "Evolution already in progress."}), 409

    thread = threading.Thread(target=background_evolution)
    thread.start()
    return jsonify({"message": "Evolution signal dispatched to background thread."})


@app.route("/logs", methods=["GET"])
def get_logs():
    log_files = ["evolution_logs.md", "sync_recovery.txt", "system_gate.log"]
    combined_logs = {}

    for log in log_files:
        if os.path.exists(log):
            with open(log, "r") as f:
                combined_logs[log] = f.readlines()[-20:]

    return jsonify(combined_logs)


@app.route("/recover", methods=["POST"])
def manual_recovery():
    state.status = "RECOVERING"
    thread = threading.Thread(target=state.recovery.run)
    thread.start()
    return jsonify({"message": "Sovereign Recovery Engine Engaged."})


@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    if state.status == "FAULTY" or state.status == "CRITICAL_FAULT":
        return jsonify({"error": "System is not healthy."}), 503
    return jsonify({"message": "System is healthy."})


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(500)
def internal_error(error):
    state.status = "FAULTY"
    return jsonify({"error": "Internal Core Collapse."}), 500


if __name__ == "__main__":
    print("""
    --- SOVEREIGN API GATEWAY ONLINE ---
    """)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
