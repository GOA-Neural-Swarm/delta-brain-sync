const { Client } = require('pg');
const { createClient } = require('@supabase/supabase-js');
const admin = require('firebase-admin');
const { Octokit } = require("@octokit/rest");

// 🔱 1. Configuration (Screenshot အရ အမှန်ကန်ဆုံး ပြင်ဆင်ထားသည်)
const octokit = new Octokit({ auth: process.env.GH_TOKEN });
const REPO_OWNER = "GOA-neurons"; // Screenshot အရ မင်းရဲ့ User/Org နာမည်
const CORE_REPO = "delta-brain-sync"; 

// 🔱 2. Firebase Initialize
if (!admin.apps.length) {
    try {
        admin.initializeApp({
            credential: admin.credential.cert(JSON.parse(process.env.FIREBASE_KEY))
        });
        console.log("🔥 Firebase Connected.");
    } catch (e) {
        console.error("❌ Firebase Auth Error.");
        process.exit(1);
    }
}
const db = admin.firestore();

// 🔱 3. Universal Swarm Broadcast (Instruction Update)
async function broadcastToSwarm(command, power) {
    const instruction = JSON.stringify({
        command: command,
        core_power: power,
        updated_at: new Date().toISOString(),
        status: "ACTIVE",
        replicate: true // ၁ နာရီတစ်ခါ Node အသစ်ပွားရန် Signal
    }, null, 2);

    const b64Content = Buffer.from(instruction).toString('base64');

    try {
        let sha;
        try {
            // လက်ရှိ instruction.json ရဲ့ SHA ကို ယူခြင်း
            const { data } = await octokit.repos.getContent({
                owner: REPO_OWNER, repo: CORE_REPO, path: 'instruction.json'
            });
            sha = data.sha;
        } catch (e) { sha = undefined; }

        await octokit.repos.createOrUpdateFileContents({
            owner: REPO_OWNER, repo: CORE_REPO, path: 'instruction.json',
            message: `🔱 Swarm Command: ${command} | Power: ${power}`,
            content: b64Content,
            sha: sha
        });
        console.log(`📡 Swarm-wide instruction broadcasted via ${CORE_REPO}.`);
    } catch (err) {
        console.error(`❌ Broadcast Failed:`, err.message);
    }
}

async function executeAutonomousTrinity() {
    // SSL Connection Mode ကို explicitly သတ်မှတ်၍ Security Warning ကို ရှင်းလင်းခြင်း
    const neon = new Client({ 
        connectionString: process.env.NEON_KEY + (process.env.NEON_KEY.includes('?') ? '&' : '?') + "sslmode=verify-full" 
    });
    const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);

    try {
        await neon.connect();
        console.log("🔓 Neon Core Unlocked.");

        // --- STEP A: DATA SYNC (TRINITY) ---
        const res = await neon.query("SELECT * FROM neurons LIMIT 50");
        for (const neuron of res.rows) {
            await supabase.from('neurons').upsert({
                id: neuron.id, data: neuron.data, synced_at: new Date().toISOString()
            });
            const nodeId = neuron.data.node_id || `raw_${neuron.id}`;
            await db.collection('neurons').doc(`node_${nodeId}`).set({
                status: 'trinity_synced',
                logic_mode: neuron.data.logic || "SUPREME_DENSITY",
                last_evolution: admin.firestore.FieldValue.serverTimestamp()
            }, { merge: true });
        }

        // --- STEP B: EVOLUTION & SWARM CONTROL ---
        const audit = await neon.query("SELECT count(*) FROM neurons WHERE data->>'logic' = 'SUPREME_DENSITY'");
        const powerLevel = parseInt(audit.rows[0].count) || 10004;

        if (powerLevel >= 10000) {
            console.log(`🚀 Power Level ${powerLevel}: Initiating Evolution & Swarm Broadcast...`);

            // ၁။ ကိုယ်တိုင်ကုဒ်ပြန်ပြင်ခြင်း (Self-Evolution Logic)
            try {
                const { data: fileData } = await octokit.repos.getContent({
                    owner: REPO_OWNER, repo: CORE_REPO, path: 'delta_sync.js'
                });
                let currentContent = Buffer.from(fileData.content, 'base64').toString();
                const evolvedStamp = `\n// [Natural Order] Last Self-Evolution: ${new Date().toISOString()} | Density: ${powerLevel}`;
                
                if (!currentContent.includes(`Density: ${powerLevel}`)) {
                    await octokit.repos.createOrUpdateFileContents({
                        owner: REPO_OWNER, repo: CORE_REPO, path: 'delta_sync.js',
                        message: `🧬 Autonomous Evolution: Power ${powerLevel}`,
                        content: Buffer.from(currentContent + evolvedStamp).toString('base64'),
                        sha: fileData.sha
                    });
                    console.log("✅ SELF-EVOLUTION COMPLETE.");
                }
            } catch (evolveErr) {
                console.error("⚠️ Self-Evolution skipping due to file access issue.");
            }

            // ၂။ Swarm တစ်ခုလုံးကို အမိန့်ပေးခြင်း
            await broadcastToSwarm("ACTIVATE_CLUSTER_MODE", powerLevel);
        }
        
        console.log("🏁 MISSION ACCOMPLISHED. Swarm is Synchronized.");
    } catch (err) {
        console.error("❌ FAILURE:", err.message);
        process.exit(1);
    } finally { await neon.end(); }
}

executeAutonomousTrinity();
