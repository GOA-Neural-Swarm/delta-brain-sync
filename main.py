import os
import logging
from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

@app.route("/api/data", methods=["GET"])
def get_data():
    data = db.session.query("SELECT * FROM table").all()
    return jsonify([dict(row) for row in data])

if __name__ == "__main__":
    app.run(debug=True)