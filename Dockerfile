# 🛸 Python High-Stability Image
FROM python:3.10-slim

# 🛠️ Install Git (Crucial for Sovereign Push)
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# 📂 Setup Workspace
WORKDIR /app

# 📦 Install Core Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 🧬 Copy System Source
COPY . .

# 🛰️ Environment Variables (Default to Headless Sovereign Mode)
ENV HEADLESS_MODE=true
ENV PYTHONUNBUFFERED=1

# 🔱 Launch Autonomous Cycle
CMD ["python", "main.py"]
