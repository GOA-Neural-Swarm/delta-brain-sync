import os
import subprocess
import sys

infra = {
    "recovery.py": [
        'import os',
        'def recover():',
        '  if os.path.exists("agi_system.db-journal"): os.remove("agi_system.db-journal")',
    ],
    "flask_api.py": [
        'from flask import Flask, jsonify',
        'app = Flask(__name__)',
        '@app.route("/api/health")',
        'def h():',
        '  return jsonify({"status": "healthy"})',
    ],
    "requirements.txt": [
        'flask',
    ],
}

def create_file(filename, content):
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            f.write("\n".join(content))

def install_dependencies(filename):
    if os.path.exists(filename):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", filename])

for k, v in infra.items():
    create_file(k, v)

install_dependencies("requirements.txt")

def update_infra(new_infra):
    for k, v in new_infra.items():
        if k in infra:
            infra[k] = v
        else:
            infra[k] = v
        create_file(k, v)

# Example usage:
new_infra = {
    "new_file.py": [
        'import os',
        'def new_function():',
        '  print("New function added")',
    ],
}
update_infra(new_infra)