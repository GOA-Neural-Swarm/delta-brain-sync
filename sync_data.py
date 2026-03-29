import psycopg2
from psycopg2 import sql
import json
import os
import sys
import logging

# Logger Setup - OMEGA System Monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | 🛰️ [DATA-SYNC] | %(message)s'
)
logger = logging.getLogger("Sovereign-Sync")

class NeonSovereignEngine:
    def __init__(self):
        # Environment keys အားလုံးကို စစ်ဆေးပြီး strip လုပ်မယ်
        raw_url = os.environ.get('NEON_DB_URL') or os.environ.get('NEON_URL') or os.environ.get('NEON_KEY')
        if not raw_url:
            logger.error("Database URL context not found in Environment Secrets.")
            sys.exit(1)
            
        self.db_url = raw_url.strip()
        if self.db_url.startswith("postgres://"):
            self.db_url = self.db_url.replace("postgres://", "postgresql://", 1)

    def connect(self):
        """ Establish a secure connection to Neon PostgreSQL """
        return psycopg2.connect(self.db_url, sslmode='verify-full')

    def self_heal_schema(self, error_msg):
        """ 
        [AUTO-MIGRATION] AI logic to fix the database schema 
        when it detects missing components.
        """
        logger.warning(f"Detecting System Inconsistency: {error_msg}")
        conn = self.connect()
        cur = conn.cursor()
        
        try:
            # ၁။ intelligence_core table မရှိရင် ဆောက်မယ်
            if "relation \"intelligence_core\" does not exist" in error_msg.lower():
                logger.info("🛠️ Creating missing table: intelligence_core...")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS intelligence_core (
                        id SERIAL PRIMARY KEY,
                        module_name TEXT UNIQUE,
                        logic_data JSONB,
                        logic_hash TEXT DEFAULT 'initial_sync',
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    INSERT INTO intelligence_core (module_name, logic_data) 
                    VALUES ('Singularity Evolution Node', '{}') 
                    ON CONFLICT DO NOTHING;
                """)
            
            # ၂။ logic_hash column မရှိရင် တိုးမယ်
            elif "column \"logic_hash\" does not exist" in error_msg.lower():
                logger.info("🧬 Injecting missing column: logic_hash...")
                cur.execute("ALTER TABLE intelligence_core ADD COLUMN IF NOT EXISTS logic_hash TEXT DEFAULT 'stable_v1';")
            
            conn.commit()
            logger.info("✅ Database Self-Healing Protocol: SUCCESS.")
        except Exception as e:
            logger.error(f"Failed to auto-heal database: {e}")
        finally:
            cur.close()
            conn.close()

    def sync(self):
        """ Execute Master Sync Protocol """
        try:
            conn = self.connect()
            cur = conn.cursor()
            
            # Fetch data with all columns including logic_hash
            query = """
                SELECT logic_data, logic_hash 
                FROM intelligence_core 
                WHERE module_name = 'Singularity Evolution Node';
            """
            cur.execute(query)
            row = cur.fetchone()
            
            if row:
                logic_data, logic_hash = row
                payload = {
                    "metadata": {
                        "source": "Neon-V4-Cloud",
                        "hash": logic_hash,
                        "status": "HYPER_EXPANSION_READY"
                    },
                    "data": logic_data
                }
                
                with open('ai_status.json', 'w') as f:
                    json.dump(payload, f, indent=4)
                logger.info(f"✅ Neural Parity Achieved. Sync Hash: {logic_hash}")
            else:
                logger.warning("Target Module not found. Initiating placeholder...")
                # Table ရှိပေမဲ့ data မရှိရင် seed လုပ်မယ်
                cur.execute("INSERT INTO intelligence_core (module_name, logic_data) VALUES ('Singularity Evolution Node', '{}');")
                conn.commit()

            cur.close()
            conn.close()

        except psycopg2.Error as e:
            # Error တက်ရင် Self-Healing ကို လှမ်းခေါ်မယ်
            error_str = str(e)
            if "does not exist" in error_str:
                self.self_heal_schema(error_str)
                # ပြင်ပြီးရင် နောက်တစ်ကြိမ် ပြန်ကြိုးစားမယ် (Recursive call)
                self.sync()
            else:
                logger.error(f"Critical Sync Failure: {error_str}")

if __name__ == "__main__":
    engine = NeonSovereignEngine()
    engine.sync()
