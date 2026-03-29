from flask_api import app

if __name__ == "__main__":
    # Wrapper for production entry
    app.run(host='0.0.0.0', port=8080)