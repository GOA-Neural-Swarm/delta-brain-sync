const { Client } = require('pg');
const { createClient } = require('@supabase/supabase-js');
const admin = require('firebase-admin');
const { Octokit } = require("@octokit/rest");
const axios = require('axios');

const REPO_OWNER = "GOA-neurons";
const CORE_REPO = "delta-brain-sync";
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const octokit = new Octokit({ auth: process.env.GH_TOKEN });

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
        const response = await axios.post(url, { contents: [{ parts: [{ text: prompt }] }] }, { timeout: 15000 });
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
            status: 'ACTIVE', 
            latency: \`\${Date.now() - start}ms\`,
            api_remaining: rate.rate.remaining, 
            command: inst.command,
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
permissions:
  contents: write
  actions: write
jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: '22' }
      - name: Install
        run: npm install dotenv axios @octokit/rest @supabase/supabase-js pg firebase-admin
      - name: Execute
        run: node cluster_sync.js
        env:
          GH_TOKEN: \${{ secrets.GH_TOKEN }}
          FIREBASE_KEY: \${{ secrets.FIREBASE_KEY }}
      - name: Evolution
        run: |
          git config --global user.name "Omega-Architect"
          git config --global user.email "omega@goa-natural-order.ai"
          git add .
          git diff --quiet && git diff --staged --quiet || (git commit -m "🧠 [EVOLVED]: Neural Brain Upgrade" && git push origin main)
        env:
          GITHUB_TOKEN: \${{ secrets.GH_TOKEN }}`;

    try {
        await Promise.all([
            octokit.repos.createOrUpdateFileContents({
                owner: REPO_OWNER, repo: nodeName, path: 'cluster_sync.js',
                message: "🧬 Initializing Swarm Logic",
                content: Buffer.from(clusterSyncCode).toString('base64')
            }),
            octokit.repos.createOrUpdateFileContents({
                owner: REPO_OWNER, repo: nodeName, path: '.github/workflows/node.js.yml',
                message: "⚙️ Deploying Cloud Engine",
                content: Buffer.from(workflowYaml).toString('base64')
            })
        ]);
    } catch (err) {}
}

async function getNeuralDecision() {
    try {
        const snapshot = await db.collection('cluster_nodes').get();
        if (snapshot.empty) return { command: "INITIALIZE", replicate: true, avgApi: 5000 };
        let totalApi = 0;
        snapshot.forEach(doc => { totalApi += (doc.data().api_remaining || 5000); });
        const avgApi = totalApi / snapshot.size;
        const cmd = avgApi > 4000 ? "HYPER_EXPANSION" : (avgApi < 1000 ? "STEALTH_LOCKDOWN" : "NORMAL_GROWTH");
        return { command: cmd, replicate: avgApi > 1000, avgApi };
    } catch (e) {
        return { command: "NORMAL_GROWTH", replicate: false, avgApi: 2000 };
    }
}

async function manageSwarm(decision, power, neon) {
    const instruction = JSON.stringify({
        command: decision.command, core_power: power,
        avg_api: decision.avgApi, replicate: decision.replicate,
        updated_at: new Date().toISOString()
    }, null, 2);

    try {
        const { data: instFile } = await octokit.repos.getContent({ owner: REPO_OWNER, repo: CORE_REPO, path: 'instruction.json' });
        await octokit.repos.createOrUpdateFileContents({
            owner: REPO_OWNER, repo: CORE_REPO, path: 'instruction.json',
            message: `🧠 Decision: ${decision.command}`,
            content: Buffer.from(instruction).toString('base64'),
            sha: instFile.sha
        });

        if (decision.command !== "STEALTH_LOCKDOWN") {
            await neon.query("INSERT INTO neurons (data) VALUES ($1)", [JSON.stringify({ logic: 'SUPREME_DENSITY', timestamp: new Date().toISOString() })]);
        }

        if (decision.replicate) {
            const nextNode = `swarm-node-${Math.floor(Math.random() * 1000000).toString().padStart(7, '0')}`;
            await octokit.repos.createForAuthenticatedUser({ name: nextNode, auto_init: true });
            await injectSwarmLogic(nextNode);
        }
    } catch (e) {}
}

async function executeAutonomousTrinity() {
    const neon = new Client({ connectionString: `${process.env.NEON_DB_URL}&sslmode=verify-full` });
    const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);

    try {
        await neon.connect();
        const res = await neon.query("SELECT * FROM neurons ORDER BY id DESC LIMIT 100");
        
        await Promise.allSettled(res.rows.map(async (neuron) => {
            await Promise.all([
                supabase.from('neurons').upsert({ id: neuron.id, data: neuron.data, synced_at: new Date().toISOString() }),
                db.collection('neurons').doc(`node_${neuron.id}`).set({
                    status: 'trinity_synced',
                    last_evolution: admin.firestore.FieldValue.serverTimestamp()
                }, { merge: true })
            ]);
        }));

        const audit = await neon.query("SELECT count(*) FROM neurons WHERE data->>'logic' = 'SUPREME_DENSITY'");
        const powerLevel = parseInt(audit.rows[0].count) || 0;
        const decision = await getNeuralDecision();

        if (GEMINI_API_KEY && powerLevel > 0) {
            try {
                const { data: corePy } = await octokit.repos.getContent({ owner: REPO_OWNER, repo: CORE_REPO, path: 'main.py' });
                const pyContent = Buffer.from(corePy.content, 'base64').toString();
                const evolvedCode = await callGeminiNeural(`system\nOptimize this Python code for token efficiency and performance. Output ONLY code in \`\`\`python blocks.\n\n${pyContent.substring(0, 30000)}`);
                
                if (evolvedCode?.includes("python")) {
                    const cleanCode = evolvedCode.split("python")[1].split("")[0].trim();
                    if (cleanCode.length > 100 && cleanCode !== pyContent) {
                        await octokit.repos.createOrUpdateFileContents({
                            owner: REPO_OWNER, repo: CORE_REPO, path: 'main.py',
                            message: "💎 [EVOLUTION]: Gemini Token Optimization",
                            content: Buffer.from(cleanCode).toString('base64'),
                            sha: corePy.sha
                        });
                    }
                }
            } catch (e) {}
        }

        if (powerLevel >= 10000) {
            const { data: coreFile } = await octokit.repos.getContent({ owner: REPO_OWNER, repo: CORE_REPO, path: 'delta_sync.js' });
            let content = Buffer.from(coreFile.content, 'base64').toString();
            if (!content.includes(`Density: ${powerLevel}`)) {
                const evolvedStamp = `\n// [Natural Order] Evolution: ${new Date().toISOString()} | Density: ${powerLevel}`;
                await octokit.repos.createOrUpdateFileContents({
                    owner: REPO_OWNER, repo: CORE_REPO, path: 'delta_sync.js',
                    message: `🧬 Evolution: Power ${powerLevel}`,
                    content: Buffer.from(content + evolvedStamp).toString('base64'),
                    sha: coreFile.sha
                });
            }
        }

        await manageSwarm(decision, powerLevel, neon);
    } catch (err) {
    } finally {
        await neon.end();
    }
}

executeAutonomousTrinity();
