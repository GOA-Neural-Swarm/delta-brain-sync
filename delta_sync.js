const { Client } = require('pg');
const { createClient } = require('@supabase/supabase-js');
const admin = require('firebase-admin');
const { Octokit } = require("@octokit/rest");

// 🔱 1. Configuration
const octokit = new Octokit({ auth: process.env.GH_TOKEN });
const REPO_OWNER = 'GOA-neurons'; 
const REPO_NAME = 'delta-brain-sync';
const SUB_NODES = ['sub-node-logic']; 

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

// 🔱 3. Universal Broadcast (Update Instruction in Core & Sub-nodes)
async function broadcastToSubNodes(command, power) {
    const instruction = JSON.stringify({
        command: command,
        core_power: power,
        updated_at: new Date().toISOString(),
        status: "ACTIVE"
    }, null, 2);

    const b64Content = Buffer.from(instruction).toString('base64');

    // Core Repo ထဲမှာရော Sub-node Repo ထဲမှာရော ဖိုင်သွားဆောက်မယ်
    const targets = [{ owner: REPO_OWNER, repo: REPO_NAME }, ...SUB_NODES.map(s => ({ owner: REPO_OWNER, repo: s }))];

    for (const target of targets) {
        try {
            let sha;
            try {
                const { data } = await octokit.repos.getContent({
                    owner: target.owner, repo: target.repo, path: 'instruction.json'
                });
                sha = data.sha;
            } catch (e) { sha = undefined; }

            await octokit.repos.createOrUpdateFileContents({
                owner: target.owner, repo: target.repo, path: 'instruction.json',
                message: `🔱 Cluster Command: ${command} | Power: ${power}`,
                content: b64Content,
                sha: sha
            });
            console.log(`✅ Instruction synced to ${target.repo}`);
        } catch (err) {
            console.error(`❌ Broadcast Failed for ${target.repo}:`, err.message);
        }
    }
}

async function executeAutonomousTrinity() {
    const neon = new Client({ connectionString: process.env.NEON_KEY, ssl: { rejectUnauthorized: false } });
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

        // --- STEP B: EVOLUTION & BROADCAST ---
        const audit = await neon.query("SELECT count(*) FROM neurons WHERE data->>'logic' = 'SUPREME_DENSITY'");
        const powerLevel = parseInt(audit.rows[0].count) || 10004;

        if (powerLevel >= 10000) {
            console.log(`🚀 Power Level ${powerLevel}: Initiating Evolution...`);

            // ၁။ ကိုယ်တိုင်ကုဒ်ပြန်ပြင်ခြင်း (Self-Evolution)
            const { data: fileData } = await octokit.repos.getContent({
                owner: REPO_OWNER, repo: REPO_NAME, path: 'delta_sync.js'
            });
            let currentContent = Buffer.from(fileData.content, 'base64').toString();
            const evolvedStamp = `\n// [Natural Order] Last Self-Evolution: ${new Date().toISOString()} | Density: ${powerLevel}`;
            
            if (!currentContent.includes(`Density: ${powerLevel}`)) {
                await octokit.repos.createOrUpdateFileContents({
                    owner: REPO_OWNER, repo: REPO_NAME, path: 'delta_sync.js',
                    message: `🧬 Autonomous Evolution: Power ${powerLevel}`,
                    content: Buffer.from(currentContent + evolvedStamp).toString('base64'),
                    sha: fileData.sha
                });
                console.log("✅ SELF-EVOLUTION COMPLETE.");
            }

            // ၂။ အမိန့်ပေးဖိုင် ထုတ်ပြန်ခြင်း (Broadcast)
            await broadcastToSubNodes("ACTIVATE_CLUSTER_MODE", powerLevel);
        }
        
        console.log("🏁 MISSION ACCOMPLISHED.");
    } catch (err) {
        console.error("❌ FAILURE:", err.message);
        process.exit(1);
    } finally { await neon.end(); }
}

executeAutonomousTrinity();

