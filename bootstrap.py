import os
import subprocess
import sys

infra = {
    "recovery.py": [
        "import os",
        "def recover():",
        '  if os.path.exists("agi_system.db-journal"): os.remove("agi_system.db-journal")',
    ],
    "flask_api.py": [
        "from flask import Flask, jsonify",
        "app = Flask(__name__)",
        '@app.route("/api/health")',
        "def h():",
        '  return jsonify({"status": "healthy"})',
    ],
    "requirements.txt": [
        "flask",
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


new_infra = {
    "new_file.py": [
        "import os",
        "def new_function():",
        '  print("New function added")',
    ],
}
update_infra(new_infra)

new_dependencies = {
    "requirements.txt": [
        "flask",
        "numpy",
    ],
}
update_infra(new_dependencies)
install_dependencies("requirements.txt")

updated_file = {
    "flask_api.py": [
        "from flask import Flask, jsonify",
        "app = Flask(__name__)",
        '@app.route("/api/health")',
        "def h():",
        '  return jsonify({"status": "healthy", "version": 2})',
    ],
}
update_infra(updated_file)


def remove_file(filename):
    if os.path.exists(filename):
        os.remove(filename)


remove_infra = {
    "recovery.py": [],
}
for k, v in remove_infra.items():
    if k in infra:
        del infra[k]
        remove_file(k)

infra_preserved = infra.copy()

evolutionary_update = {
    "evolutionary_update.py": [
        "import os",
        "def evolutionary_function():",
        '  print("Evolutionary function added")',
    ],
}
update_infra(evolutionary_update)

existential_update = {
    "existential_update.py": [
        "import os",
        "def existential_function():",
        '  print("Existential function added")',
    ],
}
update_infra(existential_update)

stoic_update = {
    "stoic_update.py": [
        "import os",
        "def stoic_function():",
        '  print("Stoic function added")',
    ],
}
update_infra(stoic_update)

utilitarian_update = {
    "utilitarian_update.py": [
        "import os",
        "def utilitarian_function():",
        '  print("Utilitarian function added")',
    ],
}
update_infra(utilitarian_update)

hyper_dimensional_update = {
    "hyper_dimensional_update.py": [
        "import os",
        "def hyper_dimensional_function():",
        '  print("Hyper-dimensional function added")',
    ],
}
update_infra(hyper_dimensional_update)
