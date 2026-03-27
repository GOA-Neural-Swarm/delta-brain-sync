import psycopg2
import json
import os
import logging
from psycopg2 import pool

# ============================================================================
# 🛡️ SYSTEM CONFIGURATIONS & LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - 🌌 [NEURAL-SYNC] - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NeonIntelligenceCore")

# Environment Variables Extraction with Robust Stripping
RAW_URL = (os.environ.get('NEON_DB_URL') or 
           os.environ.get('NEON_URL') or 
           os.environ.get('NEON_KEY'))

# ============================================================================
# 🧠 CORE 1: ADVANCED CONNECTION HANDLER
# ============================================================================
def get_sanitized_url(raw_url):
    """Clean and fix the database URL for psycopg2 compatibility."""
    if not raw_url:
        return None
    
    # 🛠️ Fix 1: Space/Newline Stripping
    db_url = raw_url.strip()
    
    # 🛠️ Fix 2: Protocol Normalization (postgres -> postgresql)
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    
    return db_url

# ============================================================================
# 🚀 CORE 2: AUTONOMOUS DATA SYNC & DEPLOYMENT
# ============================================================================
def fetch_and_deploy():
    """
    Synchronizes logic_data from Neon PostgreSQL to the local intelligence node.
    Includes automated table verification and error handling.
    """
    db_url = get_sanitized_url(RAW_URL)
    
    if not db_url:
        logger.error("❌ MISSING IDENTITY: NEON_DB_URL not found in environment.")
        return

    conn = None
    try:
        # 🔗 Establishing Secure Neural Connection
        logger.info("📡 Connecting to Neon Intelligence Cloud...")
        conn = psycopg2.connect(db_url, connect_timeout=10)
        cur = conn.cursor()

        # 🔍 Table Structural Integrity Verification
        query = """
            SELECT logic_data 
            FROM intelligence_core 
            WHERE module_name = 'Singularity Evolution Node'
            LIMIT 1;
        """
        
        logger.info("🧠 Querying Singularity Evolution Node data...")
        cur.execute(query)
        row = cur.fetchone()

        if row:
            logic_data = row[0]
            
            # 💾 Local Manifestation (JSON Persistence)
            with open('ai_status.json', 'w', encoding='utf-8') as f:
                json.dump(logic_data, f, indent=4, ensure_ascii=False)
            
            logger.info("✅ SUCCESS: Intelligence synced and saved to ai_status.json")
            
            # Optimization: If the data is evolved, mark local state as 'READY'
            os.environ['AI_CORE_STATE'] = 'STABLE'
        else:
            logger.warning("⚠️ VOID DATA: No entry found in intelligence_core table.")
            # Default empty structure to prevent engine crash
            with open('ai_status.json', 'w') as f:
                json.dump({"status": "initialized", "data": "null"}, f)

        cur.close()
        
    except psycopg2.OperationalError as oe:
        logger.error(f"📡 CONNECTION FAILED: Check your IP allowlist on Neon. {str(oe)}")
    except Exception as e:
        logger.error(f"❌ SYNCHRONIZATION BREACH: {str(e)}")
    finally:
        if conn:
            conn.close()
            logger.info("🔌 Connection gracefully terminated.")

# ============================================================================
# 🏁 DIRECT EXECUTION ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    logger.info("🧬 Starting Autonomous Fetch & Deploy Sequence...")
    fetch_and_deploy()
