const { Client } = require('pg');
const { createClient } = require('@supabase/supabase-js');
const admin = require('firebase-admin');
const { Octokit } = require("@octokit/rest");

// 🔱 1. Configuration & Security (Confirmed via Screenshot)
const octokit = new Octokit({ auth: process.env.GH_TOKEN });
const REPO_OWNER = "GOA-neurons";
const CORE_REPO = "delta-brain-sync"; 

// 🔱 2. Firebase Initialize
if (!admin.apps.length) {
    try {
        admin.initializeApp({
            credential: admin.credential.cert(JSON.parse(process.env.FIREBASE_KEY)),
            databaseURL: process.env.FIREBASE_DB_URL
        });
        console.log("🔥 Firebase Connected.");
    } catch (e) {
        console.error("❌ Firebase Auth Error.");
        process.exit(1);
    }
}
const db = admin.firestore();

// 🔱 3. Deep Injection Logic (Node အသဈမြားထဲသို့ ကုဒျမြား အလိုအလြောကျ ထည့ျသှငျးခွငျး)
async function injectSwarmLogic(nodeName) {
    console.log(`🧬 Injecting Neural Logic into ${nodeName}...`);
    
    // Cluster Node ထဲတှငျ Run မည့ျ ပငျမကုဒျ
    const clusterSyncCode = `const { Octokit } = require("@octokit/rest");
const admin = require('firebase-admin');
const axios = require('axios');
const octokit = new Octokit({ auth: process.env.GH_TOKEN });
const REPO_OWNER = "${REPO_OWNER}";
const REPO_NAME = process.env.GITHUB_REPOSITORY.split('/')[1];
if (!admin.apps.length) { admin.initializeApp({ credential: admin.credential.cert(JSON.parse(process.env.FIREBASE_KEY)) }); }
const db = admin.firestore();
async function run() {
    try {
        const start = Date.now();
        const { data: inst } = await axios.get(\`https://raw.githubusercontent.com/\${REPO_OWNER}/delta-brain-sync/main/instruction.json\`);
        const { data: rate } = await octokit.rateLimit.get();
        await db.collection('cluster_nodes').doc(REPO_NAME).set({
            status: 'ACTIVE', latency: \`\${Date.now() - start}ms\`,
            api_remaining: rate.rate.remaining, command: inst.command,
            last_ping: admin.firestore.FieldValue.serverTimestamp()
        }, { merge: true });
        if (inst.replicate) { /* Replication Logic call via Core */ }
    } catch (e) { console.log(e.message); }
}
run();`;

    // GitHub Actions Workflow ဖိုငျ (၁၅ မိနဈတဈခါ Run ရနျ)
    const workflowYaml = `name: Node Sync
on:
  schedule: [{cron: "*/15 * * * *"}]
  workflow_dispatch:
jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: {node-version: '20'}
      - run: npm install @octokit/rest firebase-admin axios
      - run: node cluster_sync.js
        env:
          GH_TOKEN: \${{ secrets.GH_TOKEN }}
          FIREBASE_KEY: \${{ secrets.FIREBASE_KEY }}`;

    try {
        // cluster_sync.js ထည့ျခွငျး
        await octokit.repos.createOrUpdateFileContents({
            owner: REPO_OWNER, repo: nodeName, path: 'cluster_sync.js',
            message: "🧬 Initializing Swarm Logic",
            content: Buffer.from(clusterSyncCode).toString('base64')
        });

        // Workflow ဖိုငျ ထည့ျခွငျး
        await octokit.repos.createOrUpdateFileContents({
            owner: REPO_OWNER, repo: nodeName, path: '.github/workflows/node.js.yml',
            message: "⚙️ Deploying Cloud Engine",
            content: Buffer.from(workflowYaml).toString('base64')
        });
        console.log(`✅ ${nodeName} is now fully autonomous.`);
    } catch (err) {
        console.error(`❌ Injection Failed for ${nodeName}:`, err.message);
    }
}

// 🔱 4. Neural Decision Engine
async function getNeuralDecision() {
    const snapshot = await db.collection('cluster_nodes').get();
    let totalApi = 0;
    let nodeCount = snapshot.size;
    if (nodeCount === 0) return { command: "INITIALIZE", replicate: true };
    snapshot.forEach(doc => { totalApi += (doc.data().api_remaining || 5000); });
    const avgApi = totalApi / nodeCount;
    let cmd = avgApi > 4000 ? "HYPER_EXPANSION" : (avgApi < 1000 ? "STEALTH_LOCKDOWN" : "NORMAL_GROWTH");
    return { command: cmd, replicate: avgApi > 1000, avgApi };
}

// 🔱 5. Swarm Broadcast & Replication
async function manageSwarm(decision, power) {
    const instruction = JSON.stringify({
        command: decision.command, core_power: power,
        avg_api: decision.avgApi, replicate: decision.replicate,
        updated_at: new Date().toISOString()
    }, null, 2);

    const { data: instFile } = await octokit.repos.getContent({ owner: REPO_OWNER, repo: CORE_REPO, path: 'instruction.json' });
    await octokit.repos.createOrUpdateFileContents({
        owner: REPO_OWNER, repo: CORE_REPO, path: 'instruction.json',
        message: `🧠 Decision: ${decision.command}`,
        content: Buffer.from(instruction).toString('base64'),
        sha: instFile.sha
    });

    if (decision.replicate) {
        const nextNode = `swarm-node-${String(Math.floor(Math.random() * 1000000)).padStart(7, '0')}`;
        try {
            await octokit.repos.createForAuthenticatedUser({ name: nextNode, auto_init: true });
            console.log(`🚀 Spawned: ${nextNode}`);
            await injectSwarmLogic(nextNode); // 🧬 Injection ဖွဈစရေနျ ခကြျခငြျးခေါျယူခွငျး
        } catch (e) { console.log("Spawn skipped or exists."); }
    }
}

// 🔱 6. Main Execution (Trinity + Evolution + Neural)
async function executeAutonomousTrinity() {
    const neon = new Client({ connectionString: process.env.NEON_KEY + "&sslmode=verify-full" });
    const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);

    try {
        await neon.connect();
        const res = await neon.query("SELECT * FROM neurons ORDER BY id DESC");
        for (const neuron of res.rows) {
            await supabase.from('neurons').upsert({ id: neuron.id, data: neuron.data, synced_at: new Date().toISOString() });
            await db.collection('neurons').doc(`node_${neuron.id}`).set({ status: 'trinity_synced', last_evolution: admin.firestore.FieldValue.serverTimestamp() }, { merge: true });
        }

        const audit = await neon.query("SELECT count(*) FROM neurons WHERE data->>'logic' = 'SUPREME_DENSITY'");
        const powerLevel = parseInt(audit.rows[0].count) || 10004;
        const decision = await getNeuralDecision();

        // Self-Evolution
        if (powerLevel >= 10000) {
            const { data: coreFile } = await octokit.repos.getContent({ owner: REPO_OWNER, repo: CORE_REPO, path: 'delta_sync.js' });
            let content = Buffer.from(coreFile.content, 'base64').toString();
            if (!content.includes(`Density: ${powerLevel}`)) {
                const evolvedStamp = `\n// [Natural Order] Last Self-Evolution: ${new Date().toISOString()} | Density: ${powerLevel}`;
                await octokit.repos.createOrUpdateFileContents({
                    owner: REPO_OWNER, repo: CORE_REPO, path: 'delta_sync.js',
                    message: `🧬 Evolution: Power ${powerLevel}`,
                    content: Buffer.from(content + evolvedStamp).toString('base64'),
                    sha: coreFile.sha
                });
            }
        }

        await manageSwarm(decision, powerLevel);
        console.log("🏁 MISSION ACCOMPLISHED.");
    } catch (err) { console.error("❌ FAILURE:", err.message); } finally { await neon.end(); }
}

executeAutonomousTrinity();


// [Natural Order] Last Self-Evolution: 2026-01-19T04:30:40.655Z | Density: 10004
