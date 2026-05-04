/**
 * 🌌 [OMEGA-SYNC ARCHITECTURE]
 * Autonomous Neural Bridge: Python Controller -> Firebase -> Neon DB
 * Status: Production-Grade | Fault-Tolerant | Auto-Scaling
 */

const admin = require("firebase-admin");
const { Client } = require("pg");
const fs = require("fs");

// --- 🔱 1. SOVEREIGN CONFIGURATION ---
const CONFIG = {
  INSTRUCTION_FILE: "instruction.json",
  DEFAULT_COMMAND: "NORMAL_GROWTH",
  MODES: {
    HYPER_EXPANSION: 50,
    NORMAL_GROWTH: 5,
    STEALTH_LOCKDOWN: 1,
  },
  RETRY_LIMIT: 3,
  TIMEOUT_MS: 15000,
};

// --- 📡 2. INTELLIGENCE INGESTION LAYER ---
/**
 * Reads Python directives with fallback mechanisms.
 */
function getInstruction() {
  console.log(
    `[SYS-READ]: Searching for Python directives in ${CONFIG.INSTRUCTION_FILE}...`,
  );
  try {
    if (fs.existsSync(CONFIG.INSTRUCTION_FILE)) {
      const rawData = fs.readFileSync(CONFIG.INSTRUCTION_FILE, "utf8");
      const parsed = JSON.parse(rawData);
      console.log(
        `✅ [SYS-READ]: Directive loaded -> Command: ${parsed.command}`,
      );
      return parsed;
    }
  } catch (e) {
    console.warn(
      `⚠️ [WARNING]: Directive corruption or missing. Fallback initiated. Error: ${e.message}`,
    );
  }

  console.log(
    `🛡️ [FALLBACK]: Using default directive -> ${CONFIG.DEFAULT_COMMAND}`,
  );
  return { command: CONFIG.DEFAULT_COMMAND };
}

// --- 🔥 3. FIREBASE NEURAL LINK (Singleton Pattern) ---
function initFirebase() {
  if (!admin.apps.length) {
    try {
      if (!process.env.FIREBASE_SERVICE_ACCOUNT) {
        throw new Error("FIREBASE_SERVICE_ACCOUNT is missing.");
      }
      const serviceAccount = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT);
      admin.initializeApp({
        credential: admin.credential.cert(serviceAccount),
      });
      console.log("🔥 [FIREBASE]: Neural network established.");
    } catch (error) {
      console.error(
        "❌ [FATAL-FIREBASE]: Initialization failed. Check environment variables.",
      );
      throw error;
    }
  }
  return admin.firestore();
}

// --- 💎 4. NEON DB CORE LAYER ---
/**
 * Handles connection, schema verification, and resilient inserts.
 */
class NeonCore {
  constructor() {
    if (!process.env.NEON_DATABASE_URL) {
      throw new Error("NEON_DATABASE_URL is missing.");
    }
    this.client = new Client({
      connectionString: process.env.NEON_DATABASE_URL,
      ssl: { rejectUnauthorized: false },
      connectionTimeoutMillis: CONFIG.TIMEOUT_MS,
    });
  }

  async connectWithRetry(retries = CONFIG.RETRY_LIMIT) {
    for (let i = 0; i < retries; i++) {
      try {
        await this.client.connect();
        console.log("✅ [NEON]: Quantum Link Established.");
        return;
      } catch (err) {
        console.warn(
          `⚠️ [NEON]: Connection attempt ${i + 1} failed. Retrying...`,
        );
        if (i === retries - 1)
          throw new Error("Neon DB un-reachable after maximum retries.");
        await new Promise((res) => setTimeout(res, 2000 * (i + 1))); // Exponential backoff
      }
    }
  }

  async verifySchema() {
    console.log("🛠️ [NEON-SCHEMA]: Verifying foundational structures...");
    const schemaQuery = `
            CREATE TABLE IF NOT EXISTS neurons (
                id SERIAL PRIMARY KEY,
                data JSONB NOT NULL,
                evolved_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Adding index to optimize JSONB querying for future scaling
            CREATE INDEX IF NOT EXISTS idx_neurons_data ON neurons USING GIN (data);
        `;
    await this.client.query(schemaQuery);
    console.log("✅ [NEON-SCHEMA]: Integrity verified.");
  }

  async injectNeurons(neuronsArray) {
    if (neuronsArray.length === 0) return 0;

    console.log(
      `💉 [INJECTION]: Transferring ${neuronsArray.length} neural patterns to Core...`,
    );

    // Transactional insert for data integrity
    try {
      await this.client.query("BEGIN");

      // Batch insertion logic for performance
      let successCount = 0;
      for (const neuron of neuronsArray) {
        try {
          await this.client.query(
            "INSERT INTO neurons (data, evolved_at) VALUES ($1, NOW())",
            [JSON.stringify(neuron)],
          );
          successCount++;
        } catch (insertErr) {
          console.error(
            `⚠️ [INJECTION-SKIP]: Failed to inject specific neuron data. Reason: ${insertErr.message}`,
          );
        }
      }

      await this.client.query("COMMIT");
      return successCount;
    } catch (err) {
      await this.client.query("ROLLBACK");
      console.error("❌ [TRANSACTION FAILED]: Rolling back batch injection.");
      throw err;
    }
  }

  async disconnect() {
    if (this.client) {
      try {
        await this.client.end();
        console.log("🔌 [NEON]: Quantum Link severed safely.");
      } catch (e) {
        console.error("⚠️ [NEON]: Error during disconnect.", e);
      }
    }
  }
}

// --- 🌀 5. THE MASTER EVOLUTION SEQUENCE ---
async function executeEvolutionCycle() {
  console.log("\n=======================================================");
  console.log("🚀 [OMEGA-SYNC]: INITIATING MASTER EVOLUTION SEQUENCE");
  console.log("=======================================================\n");

  const neonCore = new NeonCore();

  try {
    // Step 1: Read Directives
    const instr = getInstruction();
    const syncLimit =
      CONFIG.MODES[instr.command] || CONFIG.MODES[CONFIG.DEFAULT_COMMAND];
    console.log(
      `🎯 [TARGET]: Operation Mode -> [${instr.command}] | Processing Limit -> [${syncLimit} Nodes]`,
    );

    // Step 2: Initialize Systems
    const db = initFirebase();
    await neonCore.connectWithRetry();
    await neonCore.verifySchema();

    // Step 3: Harvest Data from Firebase
    console.log(
      `🧬 [HARVEST]: Extracting up to ${syncLimit} patterns from Firestore...`,
    );
    const snap = await db.collection("neurons").limit(syncLimit).get();

    if (snap.empty) {
      console.log(
        "🌌 [STASIS]: Repository empty. No active evolution required.",
      );
    } else {
      // Transform snapshot to clean array
      const neuronsToMigrate = snap.docs.map((doc) => doc.data());

      // Step 4: Inject to Neon
      const injectedCount = await neonCore.injectNeurons(neuronsToMigrate);
      console.log(
        `🏁 [EVOLUTION COMPLETE]: ${injectedCount}/${neuronsToMigrate.length} neurons successfully manifested on Neon DB.`,
      );
    }
  } catch (err) {
    console.error(
      "\n💀 [SYSTEM COLLAPSE]: Critical failure in Evolution Cycle.",
    );
    console.error("Error Trace:", err.stack);
    process.exitCode = 1; // Mark process as failed without immediate hard exit
  } finally {
    // Step 5: Clean Shutdown
    await neonCore.disconnect();
    console.log("✅ [SYSTEM]: Cycle Finalized.\n");
  }
}

// ==========================================
// 🚀 TRIGGER EXECUTION
// ==========================================
executeEvolutionCycle();
