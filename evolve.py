import os
import json
import psycopg2
import firebase_admin
from firebase_admin import credentials, firestore
import random
from datetime import datetime

# ---  CONFIG (ENVIRONMENT VARIABLES) ---
NEON_URL = os.environ.get('NEON_URL')

#  FIREBASE CONFIG (သန့်ရှင်းပြီးသား Private Key)
clean_private_key = (
    "-----BEGIN PRIVATE KEY-----\n"
    "MIIEvwIBADANBgkqhkiG9w0BAQEFAASCBKkwggSlAgEAAoIBAQClPqbuI2bmqOZn\n"
    "kZdHP/hIBAmgyO7Bgr8xL/dOAYvj7iWF/ndUHQYTnRml31n+y/+wmgSRxi9v1+j9\n"
    "d0+4d/ynsyTpGneWcy3HqahPMQp3fmg+2yKv+RYGVyOuCiM4xXhkHajySQiyuIUB\n"
    "VCzdiB4qYksB1wRZbL65uPD0yaMa6E5XKOfM7YaEjthRkHP/xLFE41ICcLcKPoPC\n"
    "f8PZe9DSYgyIxKHCO+LVWOPOp/elboeowcu8QtWzeKjyrUbLvMA1q92mfMJmHSdI\n"
    "Vg/oXPKzheTSjgO+VOfF221dtUOqWkzLa5eABTHQA13ONPvDVEgQ97RvjIhc0Wko\n"
    "IKEVK0bLAgMBAAECggEADcaQ1QZ3hCAtgRHWmjZ/hMVtZg2KNfCn7rpQdBzV5C0M\n"
    "zMRff1AiGw18P2NE1eR8zuSwH9T1TG4j+slxCHBcTEC3gYVW1eCJPv1qThvJAxgz\n"
    "KZZMKH5r8yBdlZs7v3Za6IG+fWBQTNHsKKWzTc6UsTTbiu45axkRN1tvHwEWS15T\n"
    "Km6dAJbW1vlSm28nZEKiy9WRu/o7CEK+i+bYsVGz9nsAp3bfHCuOj9+QoOPBSSZ/\n"
    "9foCbN8kHoz8WLG3DRn+jNPPtqhaBXL4rGgVIyAIeG7R0T4Jr3WD6oCeO2ZfaJpK\n"
    "pKf26zg7G4qahqb12Bwz2tarigZuhNm7zcwA5/BC0QKBgQDO+oXvRvF0qGbW9Mb/\n"
    "FU9LR73Nt0tLDoTc4cJSHUOgL5SF6M+l8IxfxsP3mwbhvvk+914TLpgVMkB72xHI\n"
    "mMAeTh3e6LlnVJ+OtK9TX3p3t0AbSaIk7D3ykq1nYr7Ns/2esW53Vi70tdnwA0La\n"
    "2cvwE9HVOI79krvQFdoYYOEJ8QKBgQDMYbsa3BE6kEPDkfrr+2VY8NRhthj7IUqE\n"
    "ensdVs2+EDzTtIaWI4MyxZLQZP94NZvRcciJfd+PP+uMG8xUS3GaNHnYTAofu+hg\n"
    "QtXfON6QPIXICFSrC2K7AOC5BJA6pK6S1WLM8BRO3xbk3chhMihoJwNSMPiAxbcC\n"
    "0ytBCkOAewKBgQCYxkZSJbVX/G1cQPUZl6sdz+iDjcXfsunS+Disz7j45eXlKcEL\n"
    "pRCYKWjAvQdJXeMv3Prtgbjz/FGomjz4Kfe05sgZnwIrCUV02l2HVrRY5URGYAV0\n"
    "54OaJzYjV7mqsC6GEkWNhGnIaupgxKd2Tsi/foGltsek18gVgeunjurMoQKBgQDH\n"
    "WfxaspTLfrPaKqWJT+kG28EMncW4DjzVA3LapzR/Uu9BwDAWegUanMQbKKhW5FNb\n"
    "85QbJ//LhhmGzAZ9oijotI60f1bQpUR/wDFETgAoyB/lgNq1C6H9rVmEngLgcIkn\n"
    "B6QbKYFlfQyjqAAvbfEjxgnjPYjmcfOUec0S36P/yQKBgQCjmM5CNTaOxKMrWpkY\n"
    "Q0v1UKXqGoXNBLdUZDEIAAZtJlmV2kTvpY9bTQRbNCSSi0QScUO0TiYCeXLxqcy4\n"
    "M5racd5edE0D4xkfB0JyNP9HMda55/IHrf3HgI/6mhsN9Or1aDILitdhPLz4YHpU\n"
    "EFQXnDFmI44M2LF0c9vKlPzmGg==\n"
    "-----END PRIVATE KEY-----\n"
)

def start_secure_evolution():
    print(f" HARDENED ENGINE START: {datetime.now()}")
    
    if not NEON_URL:
        print(" ERROR: NEON_URL is missing! Check GitHub Secrets.")
        return

    try:
        # 1. Neon Database Connection
        conn = psycopg2.connect(NEON_URL)
        cur = conn.cursor()
        
        # Latest Gen ကို Security ပါပါနဲ့ ဖတ်မယ်
        cur.execute("SELECT (data->>'gen')::int, (data->>'bias')::float FROM neurons ORDER BY evolved_at DESC LIMIT 1;")
        row = cur.fetchone()
        
        current_gen = int(row[0]) if row and row[0] is not None else 0
        current_bias = float(row[1]) if row and row[1] is not None else 0.1
        
        #  Mutation Logic
        next_gen = current_gen + 1
        next_bias = round(current_bias + random.uniform(-0.01, 0.01), 4)
        
        print(f" EVOLVING: Gen {current_gen} -> {next_gen}")

        # 2. Database Update (Atomic Commit)
        new_data = json.dumps({"gen": next_gen, "bias": next_bias, "engine": "Real-Steel-v2"})
        cur.execute("INSERT INTO neurons (data, evolved_at) VALUES (%s, %s);", (new_data, datetime.now()))
        conn.commit()
        print(f" NEON: Gen {next_gen} recorded.")

        # 3. Firebase Sync
        if not firebase_admin._apps:
            cred = credentials.Certificate({
                "type": "service_account",
                "project_id": "april-5061f",
                "private_key": clean_private_key,
                "client_email": "firebase-adminsdk-fbsvc@april-5061f.iam.gserviceaccount.com",
                "token_uri": "https://oauth2.googleapis.com/token",
            })
            firebase_admin.initialize_app(cred)
        
        db = firestore.client()
        db.collection('evolution_stats').document('latest').set({
            'gen': next_gen,
            'bias': next_bias,
            'last_sync': firestore.SERVER_TIMESTAMP,
            'security_level': 'HARDENED'
        }, merge=True)
        print(" FIREBASE: Synced.")

    except Exception as e:
        print(f" CRITICAL FAILURE: {str(e)}")
    finally:
        if 'conn' in locals():
            cur.close()
            conn.close()

if __name__ == "__main__":
    start_secure_evolution()
