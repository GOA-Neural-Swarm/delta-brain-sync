import os
infra = {
    'recovery.py': 'import os\ndef recover():\n  if os.path.exists("agi_system.db-journal"): os.remove("agi_system.db-journal")',
    'flask_api.py': 'from flask import Flask, jsonify\napp = Flask(__name__)\n@app.route("/api/health")\ndef h(): return jsonify({"status": "healthy"})'
}
for k, v in infra.items():
    if not os.path.exists(k):
        with open(k, 'w') as f: f.write(v)
