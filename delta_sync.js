const { Client } = require('pg');
const { createClient } = require('@supabase/supabase-js');
const admin = require('firebase-admin');
const { Octokit } = require("@octokit/rest");

// 🔱 1. Configuration & Security
const octokit = new Octokit({ auth: process.env.GH_TOKEN });
const REPO_OWNER = "GOA-neurons"; //
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

// 🔱 3. Neural Decision Engine (Sub-node Data များကို ခွဲခြမ်းစိတ်ဖြာခြင်း)
async function getNeuralDecision() {
    console.log("🧠 Core Neural Engine Analyzing Swarm Intelligence...");
    const snapshot = await db.collection('cluster_nodes').get();
    
    let totalApiRemaining = 0;
    let nodeCount = snapshot.size;

    if (nodeCount === 0) return { command: "INITIALIZE", replicate: true };

    snapshot.forEach(doc => {
        totalApiRemaining += (doc.data().api_remaining || 5000);
    });

    const avgApi = totalApiRemaining / nodeCount;
    let finalCommand = "STABILIZE";
    let replicateMode = false;

    // API ကျန်းမာရေးပေါ်မူတည်၍ Decision ချမှတ်ခြင်း
    if (avgApi > 4000) {
        finalCommand = "HYPER_EXPANSION"; 
        replicateMode = true;
    } else if (avgApi < 1000) {
        finalCommand = "STEALTH_LOCKDOWN"; 
        replicateMode = false;
    } else {
        finalCommand = "NORMAL_GROWTH";
        replicateMode = true;
    }

    return { command: finalCommand, replicate: replicateMode, avgApi };
}

// 🔱 4. Universal Swarm Broadcast (Instruction Update)
async function broadcastToSwarm(decision, power) {
    const instruction = JSON.stringify({
        command: decision.command,
        core_power: power,
        avg_swarm_api: decision.avgApi,
        replicate: decision.replicate,
        updated_at: new Date().toISOString(),
        status: "ACTIVE"
    }, null, 2);

    try {
        const { data } = await octokit.repos.getContent({
            owner: REPO_OWNER, repo: CORE_REPO, path: 'instruction.json'
        });
        
        await octokit.repos.createOrUpdateFileContents({
            owner: REPO_OWNER, repo: CORE_REPO, path: 'instruction.json',
            message: `🧠 Neural Decision: ${decision.command} | Power: ${power}`,
            content: Buffer.from(instruction).toString('base64'),
            sha: data.sha
        });
        console.log(`📡 Broadcasted: ${decision.command} to the Swarm.`);
    } catch (err) {
        console.error(`❌ Broadcast Failed:`, err.message);
    }
}

// 🔱 5. Autonomous Trinity Execution
async function executeAutonomousTrinity() {
    const neon = new Client({ 
        connectionString: process.env.NEON_KEY + (process.env.NEON_KEY.includes('?') ? '&' : '?') + "sslmode=verify-full" 
    });
    const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);

    try {
        await neon.connect();
        console.log("🔓 Neon Core Unlocked.");

        // --- STEP A: DATA SYNC (TRINITY LOGIC) ---
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

        // --- STEP B: NEURAL ANALYSIS & DECISION ---
        const audit = await neon.query("SELECT count(*) FROM neurons WHERE data->>'logic' = 'SUPREME_DENSITY'");
        const powerLevel = parseInt(audit.rows[0].count) || 10004;
        const decision = await getNeuralDecision();

        // --- STEP C: SELF-EVOLUTION (ကိုယ်တိုင်ကုဒ်ပြင်ခြင်း) ---
        if (powerLevel >= 10000) {
            console.log(`🚀 Power Level ${powerLevel}: Triggering Evolution...`);
            try {
                const { data: fileData } = await octokit.repos.getContent({
                    owner: REPO_OWNER, repo: CORE_REPO, path: 'delta_sync.js'
                });
                let currentContent = Buffer.from(fileData.content, 'base64').toString();
                const evolvedStamp = `\n// [Natural Order] Last Self-Evolution: ${new Date().toISOString()} | Density: ${powerLevel} | Decision: ${decision.command}`;
                
                if (!currentContent.includes(`Density: ${powerLevel}`)) {
                    await octokit.repos.createOrUpdateFileContents({
                        owner: REPO_OWNER, repo: CORE_REPO, path: 'delta_sync.js',
                        message: `🧬 Autonomous Evolution: Power ${powerLevel}`,
                        content: Buffer.from(currentContent + evolvedStamp).toString('base64'),
                        sha: fileData.sha
                    });
                    console.log("✅ SELF-EVOLUTION COMPLETE.");
                }
            } catch (evolveErr) { console.error("⚠️ Evolution Access Issue."); }
        }

        // --- STEP D: SWARM CONTROL ---
        await broadcastToSwarm(decision, powerLevel);
        
        console.log("🏁 MISSION ACCOMPLISHED. Neural Swarm Synchronized.");
    } catch (err) {
        console.error("❌ FAILURE:", err.message);
        process.exit(1);
    } finally { await neon.end(); }
}

executeAutonomousTrinity();

// [Natural Order] Last Self-Evolution: 2026-01-19T04:15:00.000Z | Density: 10004 | Decision: HYPER_EXPANSION
