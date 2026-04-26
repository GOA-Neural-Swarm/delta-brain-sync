import os
import subprocess
import sys
import time
import json
import traceback
import re
import random
import base64
import math
from collections import deque

# Phase 1: Install/Verify all dependencies
required_packages = [
    "transformers>=4.27.0",
    "bitsandbytes",
    "accelerate",
    "firebase-admin",
    "GitPython",
    "scikit-learn",
    "numpy",
    "torch",
    "requests",
]


def install_packages(packages):
    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                *packages,
                "--quiet",
                "--no-cache-dir",
            ]
        )
        print("✅ [SYSTEM]: Dependencies verified and installed.")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Install Warning: Error installing requirements: {e}")


install_packages(required_packages)

# Phase 2: Import third-party libraries after installation
import numpy as np
import torch
import requests
import git
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import firebase_admin
from firebase_admin import credentials, db

# Phase 3: Safe Firebase Initialization
if not firebase_admin._apps:
    try:
        # Replace with your actual credentials and database URL
        # cred = credentials.Certificate('path/to/serviceAccountKey.json')
        # firebase_admin.initialize_app(cred, {'databaseURL': 'https://your-db.firebaseio.com/'})
        pass
    except Exception as e:
        print(f"⚠️ Firebase Init Warning: {e}")

# ... rest of your code ...
