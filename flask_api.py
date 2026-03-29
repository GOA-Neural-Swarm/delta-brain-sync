from flask import Flask, jsonify
from brain import NeuralCore

app = Flask(__name__)
core = NeuralCore(generation=1)

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({
        "gen": core.gen,
        "sync_state": core.sync_state,
        "neural_load": core.neural_load
    })

@app.route('/evolve', methods=['POST'])
def trigger_evolution():
    core._trigger_neural_expansion()
    return jsonify({"message": "Evolution signal dispatched."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)