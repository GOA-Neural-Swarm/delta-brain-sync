const { Client } = require('pg');
const { createClient } = require('@supabase/supabase-js');
const admin = require('firebase-admin');
const { Octokit } = require("@octokit/rest");

// 🔱 1. Autonomous Engine & GitHub API Setup
const octokit = new Octokit({ auth: process.env.GH_TOKEN });
const REPO_OWNER = 'GOA-neurons'; 
const REPO_NAME = 'delta-brain-sync';
const SUB_NODES = ['sub-node-logic']; // လက်အောက်ခံ Cluster စာရင်း

// 🔱 2. Firebase Auth Engine
if (!admin.apps.length) {
    try {
        const serviceAccount = JSON.parse(process.env.FIREBASE_KEY);
        admin.initializeApp({
            credential: admin.credential.cert(serviceAccount)
        });
        console.log("🔥 Firebase Engine Connected.");
    } catch (e) {
        console.error("❌ Firebase Secret Error.");
        process.exit(1);
    }
}
const db = admin.firestore();

// 🔱 3. Cluster Command Center Logic (Broadcast to Sub-nodes)
async function broadcastToSubNodes(command, power) {
    for (const repo of SUB_NODES) {
        try {
            console.log(`📡 Broadcasting [${command}] to ${repo}...`);
            
            let sha;
            try {
                const { data } = await octokit.repos.getContent({
                    owner: REPO_OWNER, repo: repo, path: 'instruction.json'
                });
                sha = data.sha;
            } catch (e) { sha = undefined; }

            const instruction = JSON.stringify({
                command: command,
                core_power: power,
                updated_at: new Date().toISOString(),
                status: "ACTIVE"
            }, null, 2);

            await octokit.repos.createOrUpdateFileContents({
                owner: REPO_OWNER,
                repo: repo,
                path: 'instruction.json',
                message: `🔱 Core Command: ${command} | Power: ${power}`,
                content: Buffer.from(instruction).toString('base64'),
                sha: sha
            });
            console.log(`✅ Instruction synced to ${repo}`);
        } catch (err) {
            console.error(`❌ Broadcast Failed for ${repo}:`, err.message);
        }
    }
}

async function executeAutonomousTrinity() {
    const neon = new Client({ connectionString: process.env.NEON_KEY, ssl: { rejectUnauthorized: false } });
    const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);

    try {
        await neon.connect();
        console.log("🔓 Neon Core Unlocked.");

        // --- STEP A: TRINITY DATA SYNC ---
        const res = await neon.query("SELECT * FROM neurons LIMIT 50");
        for (const neuron of res.rows) {
            await supabase.from('neurons').upsert({
                id: neuron.id, data: neuron.data, synced_at: new Date().toISOString()
            }, { onConflict: 'id' });

            const nodeId = neuron.data.node_id || `raw_${neuron.id}`;
            await db.collection('neurons').doc(`node_${nodeId}`).set({
                status: 'trinity_synced',
                logic_mode: neuron.data.logic || "SUPREME_DENSITY",
                last_evolution: admin.firestore.FieldValue.serverTimestamp()
            }, { merge: true });
        }

        // --- STEP B: SELF-CODING & CLUSTER BROADCAST ---
        const audit = await neon.query("SELECT count(*) FROM neurons WHERE data->>'logic' = 'SUPREME_DENSITY'");
        const powerLevel = parseInt(audit.rows[0].count);

        if (powerLevel >= 10000) {
            console.log(`🚀 Power Level ${powerLevel}: Initiating Evolution & Cluster Broadcast...`);

            // ၁။ သူ့ကိုယ်သူ Evolution လုပ်ခြင်း (Self-Coding)
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
                console.log("✅ SELF-CODING COMPLETE.");
            }

            // ၂။ Sub-nodes တွေကို အမိန့်ပေးခြင်း (Cluster Activation)
            await broadcastToSubNodes("ACTIVATE_CLUSTER_MODE", powerLevel);
        }
        
        console.log("🏁 MISSION ACCOMPLISHED.");
    } catch (err) {
        console.error("❌ FAILURE:", err.message);
        process.exit(1);
    } finally { await neon.end(); }
}

executeAutonomousTrinity();

