import telemetry_bridge
import psycopg2
from psycopg2 import sql
import json
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
logging.basicConfig(level=logging.INFO, format='%(asctime)s | [DATA-SYNC] | %(message)s', handlers=[logging.StreamHandler(), RotatingFileHandler('sovereign_sync.log', maxBytes=1000000, backupCount=5)])
logger = logging.getLogger('Sovereign-Sync')

class NeonSovereignEngine:

    def __init__(self):
        raw_url = os.environ.get('NEON_DB_URL') or os.environ.get('NEON_URL') or os.environ.get('NEON_KEY')
        if not raw_url:
            logger.error('Database URL context not found in Environment Secrets.')
            sys.exit(1)
        self.db_url = raw_url.strip()
        if self.db_url.startswith('postgres://'):
            self.db_url = self.db_url.replace('postgres://', 'postgresql://', 1)

    def connect(self):
        """
        Establishes a high-performance, secure connection to the Neon PostgreSQL cluster.
        Optimized for cloud-native environments (GitHub Actions) with resilient SSL
        and TCP keepalive configurations.
        """
        try:
            return psycopg2.connect(self.db_url, sslmode='require', connect_timeout=10, keepalives=1, keepalives_idle=30, keepalives_interval=10, keepalives_count=5, application_name='Sovereign_Evolution_Engine')
        except psycopg2.OperationalError as e:
            logger.error(f'Critical Connection Failure: {e}')
            raise

    def self_heal_schema(self, error_msg):
        """
        [AUTO-MIGRATION] AI logic to fix the database schema
        when it detects missing components.
        """
        logger.warning(f'Detecting System Inconsistency: {error_msg}')
        conn = self.connect()
        cur = conn.cursor()
        try:
            if 'relation "intelligence_core" does not exist' in error_msg.lower():
                logger.info('Creating missing table: intelligence_core...')
                cur.execute("\n                    CREATE TABLE IF NOT EXISTS intelligence_core (\n                        id SERIAL PRIMARY KEY,\n                        module_name TEXT UNIQUE,\n                        logic_data JSONB,\n                        logic_hash TEXT DEFAULT 'initial_sync',\n                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n                    );\n                    INSERT INTO intelligence_core (module_name, logic_data)\n                    VALUES ('Singularity Evolution Node', '{}')\n                    ON CONFLICT DO NOTHING;\n                ")
            elif 'column "logic_hash" does not exist' in error_msg.lower():
                logger.info('Injecting missing column: logic_hash...')
                cur.execute("ALTER TABLE intelligence_core ADD COLUMN IF NOT EXISTS logic_hash TEXT DEFAULT 'stable_v1';")
            conn.commit()
            logger.info('Database Self-Healing Protocol: SUCCESS.')
        except Exception as e:
            logger.error(f'Failed to auto-heal database: {e}')
        finally:
            cur.close()
            conn.close()

    def sync(self):
        """Execute Master Sync Protocol"""
        try:
            conn = self.connect()
            cur = conn.cursor()
            query = "\n                SELECT logic_data, logic_hash\n                FROM intelligence_core\n                WHERE module_name = 'Singularity Evolution Node';\n            "
            cur.execute(query)
            row = cur.fetchone()
            if row:
                logic_data, logic_hash = row
                payload = {'metadata': {'source': 'Neon-V4-Cloud', 'hash': logic_hash, 'status': 'HYPER_EXPANSION_READY'}, 'data': logic_data}
                with open('ai_status.json', 'w') as f:
                    json.dump(payload, f, indent=4)
                logger.info(f'Neural Parity Achieved. Sync Hash: {logic_hash}')
            else:
                logger.warning('Target Module not found. Initiating placeholder...')
                cur.execute("INSERT INTO intelligence_core (module_name, logic_data) VALUES ('Singularity Evolution Node', '{}');")
                conn.commit()
            cur.close()
            conn.close()
        except psycopg2.Error as e:
            error_str = str(e)
            if 'does not exist' in error_str:
                self.self_heal_schema(error_str)
                self.sync()
            else:
                logger.error(f'Critical Sync Failure: {error_str}')
if __name__ == '__main__':
    engine = NeonSovereignEngine()
    engine.sync()