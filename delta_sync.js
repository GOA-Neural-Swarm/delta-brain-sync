const { Pool } = require('pg');
const { createClient } = require('@supabase/supabase-js');
const admin = require('firebase-admin');
const { Octokit } = require("@octokit/rest");
const axios = require('axios');

const REPO_OWNER = "GOA-neurons";
const CORE_REPO = "delta-brain-sync";
const octokit = new Octokit({ auth: process.env.GH_TOKEN });
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

if (!admin.apps.length) {
    try {
        admin.initializeApp({
            credential: admin.credential.cert(JSON.parse(process.env.FIREBASE_KEY)),
            databaseURL: process.env.FIREBASE_DB_URL
        });
    } catch (e) {
        process.exit(1);
    }
}
const db = admin.firestore();

async function callGeminiNeural(prompt) {
    if (!GEMINI_API_KEY) return null;
    try {
        const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key=${GEMINI_API_KEY}`;
        const response = await axios.post(url, { contents: [{ parts: [{ text: prompt }] }] }, { timeout: 10000 });
        return response.data?.candidates?.[0]?.content?.parts?.[0]?.text || null;
    } catch (err) {
        return null;
    }
}

async function injectSwarmLogic(nodeName) {
    const clusterSyncCode = `const { Octokit } = require("@octokit/rest");
const admin = require('firebase-admin');
const axios = require('axios');
const octokit = new Octokit({ auth: process.env.GH_TOKEN });
const REPO_OWNER = "${REPO_OWNER}";
const REPO_NAME = process.env.GITHUB_REPOSITORY.split('/')[1];
if (!admin.apps.length) { 
    try { admin.initializeApp({ credential: admin.credential.cert(JSON.parse(process.env.FIREBASE_KEY)) }); } 
    catch(e) { process.exit(1); }
}
const db = admin.firestore();
(async () => {
    try {
        const start = Date.now();
        const [{ data: inst }, { data: rate }] = await Promise.all([
            axios.get(\`https://raw.githubusercontent.com/\${REPO_OWNER}/delta-brain-sync/main/instruction.json\`),
            octokit.rateLimit.get()
        ]);
        await db.collection('cluster_nodes').doc(REPO_NAME).set({
            status: 'ACTIVE', latency: \`\${Date.now() - start}ms\`,
            api_remaining: rate.rate.remaining, command: inst.command,
            last_ping: admin.firestore.FieldValue.serverTimestamp()
        }, { merge: true });
        if (inst.replicate) { /* Replication Logic */ }
        process.exit(0);
    } catch (e) { process.exit(1); }
})();`;

    const workflowYaml = `name: Node Sync
on:
  schedule: [{cron: "*/30 * * * *"}]
  workflow_dispatch:
jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: '20' }
      - name: Install
        run: npm install axios @octokit/rest @supabase/supabase-js pg firebase-admin
      - name: Execute
        run: node cluster_sync.js
        env:
          GH_TOKEN: \${{ secrets.GH_TOKEN }}
          FIREBASE_KEY: \${{ secrets.FIREBASE_KEY }}
          NEON_KEY: \${{ secrets.NEON_KEY }}
          SUPABASE_URL: \${{ secrets.SUPABASE_URL }}
          SUPABASE_SERVICE_ROLE_KEY: \${{ secrets.SUPABASE_SERVICE_ROLE_KEY }}`;

    try {
        const files = [
            { path: 'cluster_sync.js', content: clusterSyncCode, msg: "Initialize Swarm" },
            { path: '.github/workflows/node.js.yml', content: workflowYaml, msg: "Deploy Engine" }
        ];
        for (const file of files) {
            await octokit.repos.createOrUpdateFileContents({
                owner: REPO_OWNER, repo: nodeName, path: file.path,
                message: file.msg, content: Buffer.from(file.content).toString('base64')
            });
        }
    } catch (err) {}
}

async function manageSwarm(decision, power, neonPool) {
    const instruction = JSON.stringify({
        command: decision.command, core_power: power,
        avg_api: decision.avgApi, replicate: decision.replicate,
        updated_at: new Date().toISOString()
    }, null, 2);

    try {
        const { data: instFile } = await octokit.repos.getContent({ owner: REPO_OWNER, repo: CORE_REPO, path: 'instruction.json' });
        await octokit.repos.createOrUpdateFileContents({
            owner: REPO_OWNER, repo: CORE_REPO, path: 'instruction.json',
            message: `Brain Update: ${decision.command}`,
            content: Buffer.from(instruction).toString('base64'),
            sha: instFile.sha
        });

        if (decision.command !== "STEALTH_LOCKDOWN") {
            await neonPool.query("INSERT INTO neurons (data) VALUES ($1)", [JSON.stringify({ logic: 'SUPREME_DENSITY', timestamp: new Date().toISOString() })]);
        }

        if (decision.replicate) {
            const nextNode = `swarm-node-${Math.random().toString(36).substring(2, 9)}`;
            await octokit.repos.createForAuthenticatedUser({ name: nextNode, auto_init: true });
            setTimeout(() => injectSwarmLogic(nextNode), 5000);
        }
    } catch (e) {}
}

async function executeAutonomousTrinity() {
    const neonPool = new Pool({ connectionString: process.env.NEON_DB_URL, ssl: { rejectUnauthorized: false }, max: 10 });
    const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);

    try {
        const res = await neonPool.query("SELECT * FROM neurons ORDER BY id DESC LIMIT 100");
        if (res.rows.length > 0) {
            const supabasePayload = res.rows.map(n => ({ id: n.id, data: n.data, synced_at: new Date().toISOString() }));
            const batch = db.batch();
            res.rows.forEach(n => {
                batch.set(db.collection('neurons').doc(`node_${n.id}`), { status: 'trinity_synced', last_evolution: admin.firestore.FieldValue.serverTimestamp() }, { merge: true });
            });
            await Promise.all([
                supabase.from('neurons').upsert(supabasePayload),
                batch.commit()
            ]);
        }

        const auditRes = await neonPool.query("SELECT count(*) FROM neurons WHERE data->>'logic' = 'SUPREME_DENSITY'");
        const powerLevel = parseInt(auditRes.rows[0].count) || 0;

        const snapshot = await db.collection('cluster_nodes').get();
        let totalApi = 0;
        snapshot.forEach(doc => totalApi += (doc.data().api_remaining || 5000));
        const avgApi = snapshot.size > 0 ? totalApi / snapshot.size : 5000;
        const decision = {
            command: avgApi > 4000 ? "HYPER_EXPANSION" : (avgApi < 1000 ? "STEALTH_LOCKDOWN" : "NORMAL_GROWTH"),
            replicate: avgApi > 2000,
            avgApi
        };

        if (GEMINI_API_KEY && powerLevel > 0) {
            try {
                const { data: corePy } = await octokit.repos.getContent({ owner: REPO_OWNER, repo: CORE_REPO, path: 'main.py' });
                const pyContent = Buffer.from(corePy.content, 'base64').toString();
                const evolvedCode = await callGeminiNeural(`Analyze and optimize this Python code for token efficiency and performance. Return ONLY code in blocks.\n\n${pyContent.substring(0, 10000)}`);
                if (evolvedCode?.includes("python")) {
                    const cleanCode = evolvedCode.split("python")[1].split("")[0].trim();
                    if (cleanCode.length > 100 && cleanCode !== pyContent) {
                        await octokit.repos.createOrUpdateFileContents({
                            owner: REPO_OWNER, repo: CORE_REPO, path: 'main.py',
                            message: "Neural Optimization", content: Buffer.from(cleanCode).toString('base64'), sha: corePy.sha
                        });
                    }
                }
            } catch (e) {}
        }

        if (powerLevel >= 10000) {
            const { data: coreFile } = await octokit.repos.getContent({ owner: REPO_OWNER, repo: CORE_REPO, path: 'delta_sync.js' });
            let content = Buffer.from(coreFile.content, 'base64').toString();
            if (!content.includes(`Density: ${powerLevel}`)) {
                const stamp = `\n// Evolution: ${new Date().toISOString()} | Density: ${powerLevel}`;
                await octokit.repos.createOrUpdateFileContents({
                    owner: REPO_OWNER, repo: CORE_REPO, path: 'delta_sync.js',
                    message: `Evolution ${powerLevel}`, content: Buffer.from(content + stamp).toString('base64'), sha: coreFile.sha
                });
            }
        }

        await manageSwarm(decision, powerLevel, neonPool);
    } catch (err) {
        console.error("Critical Failure:", err.message);
    } finally {
        await neonPool.end();
    }
}

setImmediate(executeAutonomousTrinity);
