import telemetry_bridge
import logging
import os
import sys
import threading
import time
from flask import Flask, jsonify, request
from datetime import datetime
from brain import SovereignArchitect
from recovery import SovereignRecovery
from telemetry_bridge import TelemetryBridge
from flask_cors import CORS
from flask_httpauth import HTTPTokenAuth
from werkzeug.security import generate_password_hash, check_password_hash
app = Flask(__name__)
CORS(app)
auth = HTTPTokenAuth(scheme='Bearer')
users = {'admin': generate_password_hash('password')}

@auth.verify_token
def verify_token(token):
    """Verify the authentication token"""
    for user, password in users.items():
        if check_password_hash(password, token):
            return user

class ASI_State:
    """Represents the state of the ASI system"""

    def __init__(self):
        """Initialize the ASI state"""
        self.architect = SovereignArchitect()
        self.recovery = SovereignRecovery()
        self.telemetry_bridge = TelemetryBridge()
        self.is_training = False
        self.boot_time = datetime.now()
        self.last_evolution = None
        self.neural_load = 0.0
        self.status = 'STABLE'
        self.evolution_count = 0
        self.sync_count = 0
        self.lock = threading.Lock()

    def sync(self):
        """Perform a sync operation"""
        with self.lock:
            self.sync_count += 1
            logging.info(f'Sync operation {self.sync_count} initiated...')
            try:
                self.architect.sync()
                self.recovery.sync()
                self.telemetry_bridge.sync()
                logging.info('Sync operation successful.')
            except Exception as e:
                logging.error(f'Sync error: {e}')

    def evolve(self):
        """Perform an evolution operation"""
        with self.lock:
            self.is_training = True
            self.status = 'EVOLVING'
            try:
                logging.info('Neural Expansion Sequence Initiated...')
                self.architect.execute_evolution_step()
                self.last_evolution = datetime.now()
                self.status = 'STABLE'
                self.evolution_count += 1
                logging.info('Evolution successful.')
            except Exception as e:
                self.status = 'CRITICAL_FAULT'
                logging.error(f'Evolution Error: {e}')
            finally:
                self.is_training = False

    def get_status(self):
        """Get the current status of the ASI system"""
        return {'gen': self.architect.gen, 'neural_load': f'{self.neural_load}%', 'is_training': self.is_training, 'classifier_type': self.architect.brain.classifier_type, 'last_sync': time.strftime('%Y-%m-%d %H:%M:%S'), 'status': self.status, 'evolution_count': self.evolution_count, 'sync_count': self.sync_count}
state = ASI_State()

@app.route('/', methods=['GET'])
@auth.login_required
def index():
    """Index endpoint"""
    return jsonify({'system': 'OMEGA-ASI SOVEREIGN CORE', 'version': 'X10.2.2', 'uptime': str(datetime.now() - state.boot_time), 'endpoints': ['/status', '/evolve', '/train', '/logs', '/recover', '/healthcheck', '/shutdown', '/sync']})

@app.route('/status', methods=['GET'])
@auth.login_required
def get_status():
    """Get the current status of the ASI system"""
    return jsonify(state.get_status())

@app.route('/evolve', methods=['POST'])
@auth.login_required
def trigger_evolution():
    """Trigger an evolution operation"""
    if state.is_training:
        return (jsonify({'error': 'Evolution already in progress.'}), 409)
    thread = threading.Thread(target=state.evolve)
    thread.start()
    return jsonify({'message': 'Evolution signal dispatched to background thread.'})

@app.route('/logs', methods=['GET'])
@auth.login_required
def get_logs():
    """Get the latest logs"""
    log_files = ['evolution_logs.md', 'sync_recovery.txt', 'system_gate.log']
    combined_logs = {}
    for log in log_files:
        if os.path.exists(log):
            with open(log, 'r') as f:
                combined_logs[log] = f.readlines()[-20:]
    return jsonify(combined_logs)

@app.route('/recover', methods=['POST'])
@auth.login_required
def manual_recovery():
    """Perform a manual recovery operation"""
    state.status = 'RECOVERING'
    thread = threading.Thread(target=state.recovery.run)
    thread.start()
    return jsonify({'message': 'Sovereign Recovery Engine Engaged.'})

@app.route('/healthcheck', methods=['GET'])
@auth.login_required
def healthcheck():
    """Perform a health check"""
    if state.status in ['FAULTY', 'CRITICAL_FAULT']:
        return (jsonify({'error': 'System is not healthy.'}), 503)
    return jsonify({'message': 'System is healthy.'})

@app.route('/shutdown', methods=['POST'])
@auth.login_required
def shutdown():
    """Shutdown the system"""
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    return jsonify({'message': 'Server shutting down...'})

@app.route('/sync', methods=['POST'])
@auth.login_required
def sync():
    """Perform a sync operation"""
    thread = threading.Thread(target=state.sync)
    thread.start()
    return jsonify({'message': 'Sync operation initiated.'})

@app.route('/login', methods=['POST'])
def login():
    """Login endpoint"""
    username = request.json.get('username')
    password = request.json.get('password')
    if username in users and check_password_hash(users[username], password):
        return jsonify({'token': password})
    return (jsonify({'error': 'Invalid credentials.'}), 401)

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return (jsonify({'error': 'Endpoint not found.'}), 404)

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    state.status = 'FAULTY'
    return (jsonify({'error': 'Internal Core Collapse.'}), 500)
if __name__ == '__main__':
    try:
        from brain import SovereignArchitect
        from recovery import SovereignRecovery
    except ImportError:
        logging.critical('Core components missing. Emergency recovery required.')
        print('Core components missing. Emergency recovery required.')
        sys.exit(1)
    logging.basicConfig(filename='system_gate.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    print('\n    --- SOVEREIGN API GATEWAY ONLINE ---\n    ')
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)