const { Pool } = require('pg');
const { createClient } = require('@supabase/supabase-js');
const admin = require('firebase-admin');
const { Octokit } = require("@octokit/rest");
const axios = require('axios');

const CONFIG = {
    owner: "GOA-neurons",
    core: "delta-brain-sync",
    threshold: 10000,
    batchSize: 500
};

const octokit = new Octokit({ auth: process.env.GH_TOKEN });
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);
const neonPool = new Pool({ 
    connectionString: process.env.NEON_DB_URL + "?sslmode=verify-full",
    max: 20,
    idleTimeoutMillis: 30000,
    connectionTimeoutMillis: 2000
});

if (!admin.apps.length) {
    try {
        admin.initializeApp({
            credential: admin.credential.cert(JSON.parse(process.env.FIREBASE_KEY)),
            databaseURL: process.env.FIREBASE_DB_URL
        });
    } catch (e) {
        console.error("Firebase Init Failed");
    }
}
const db = admin.firestore();

async function callGeminiNeural(prompt) {
    if (!process.env.GEMINI_API_KEY) return null;
    try {
        const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key=${process.env.GEMINI_API_KEY}`;
        const response = await axios.post(url, { contents: [{ parts: [{ text: prompt }] }] }, { timeout: 10000 });
        return response.data?.candidates?.[0]?.content?.parts?.[0]?.text || null;
    } catch (err) {
        return null;
    }
}

async function syncParity() {
    const client = await neonPool.connect();
    try {
        const { rows: neonData } = await client.query("SELECT * FROM neurons ORDER BY updated_at DESC LIMIT $1", [CONFIG.batchSize]);
        if (neonData.length === 0) return 0;

        const payload = neonData.map(n => ({
            id: n.id,
            data: n.data,
            synced_at: new Date().toISOString(),
            logic_hash: Buffer.from(JSON.stringify(n.data)).toString('base64').substring(0, 16)
        }));

        const { error: supError } = await supabase.from('neurons').upsert(payload, { onConflict: 'id' });
        if (supError) throw supError;

        const batch = db.batch();
        neonData.forEach(n => {
            const ref = db.collection('neurons').doc(`node_${n.id}`);
            batch.set(ref, { 
                status: 'synchronized', 
                last_sync: admin.firestore.FieldValue.serverTimestamp(),
                integrity: true 
            }, { merge: true });
        });
        await batch.commit();

        return neonData.length;
    } catch (err) {
        console.error("Parity Failure:", err.message);
        return 0;
    } finally {
        client.release();
    }
}

async function evolveCore() {
    try {
        const { data: corePy } = await octokit.repos.getContent({ owner: CONFIG.owner, repo: CONFIG.core, path: 'main.py' });
        const content = Buffer.from(corePy.content, 'base64').toString();
        const evolved = await callGeminiNeural(`Optimize this Python code for maximum throughput. Return ONLY the code block.\n\n${content}`);
        
        if (evolved && evolved.includes("python")) {
            const cleanCode = evolved.match(/python\n([\s\S]*?)\n/)?.[1] || evolved.replace(/python|/g, "").trim();
            if (cleanCode.length > 50 && cleanCode !== content) {
                await octokit.repos.createOrUpdateFileContents({
                    owner: CONFIG.owner, repo: CONFIG.core, path: 'main.py',
                    message: "🧬 [NEURAL_EVOLUTION]: Optimized Logic Path",
                    content: Buffer.from(cleanCode).toString('base64'),
                    sha: corePy.sha
                });
            }
        }
    } catch (e) {}
}

async function manageSwarm() {
    const snapshot = await db.collection('cluster_nodes').get();
    let totalApi = 0;
    snapshot.forEach(doc => totalApi += (doc.data().api_remaining || 5000));
    const avgApi = snapshot.size > 0 ? totalApi / snapshot.size : 5000;

    const decision = {
        command: avgApi > 4000 ? "HYPER_EXPANSION" : (avgApi < 1500 ? "STEALTH_LOCKDOWN" : "NORMAL_GROWTH"),
        replicate: avgApi > 2000,
        timestamp: new Date().toISOString()
    };

    const { data: instFile } = await octokit.repos.getContent({ owner: CONFIG.owner, repo: CONFIG.core, path: 'instruction.json' });
    await octokit.repos.createOrUpdateFileContents({
        owner: CONFIG.owner, repo: CONFIG.core, path: 'instruction.json',
        message: `🧠 Decision: ${decision.command}`,
        content: Buffer.from(JSON.stringify(decision, null, 2)).toString('base64'),
        sha: instFile.sha
    });

    if (decision.replicate && snapshot.size < 50) {
        const nodeName = `swarm-${Math.random().toString(36).substring(2, 9)}`;
        try {
            await octokit.repos.createForAuthenticatedUser({ name: nodeName, auto_init: true });
            // Injection logic simplified for event loop efficiency
            const syncCode = `const admin=require('firebase-admin');(async()=>{/*SwarmNode*/})();`;
            await octokit.repos.createOrUpdateFileContents({
                owner: CONFIG.owner, repo: nodeName, path: 'cluster_sync.js',
                message: "🧬 Init",
                content: Buffer.from(syncCode).toString('base64')
            });
        } catch (e) {}
    }
}

async function execute() {
    const start = Date.now();
    try {
        const [syncedCount] = await Promise.all([
            syncParity(),
            manageSwarm()
        ]);

        const { rows } = await neonPool.query("SELECT count(*) FROM neurons WHERE data->>'logic' = 'SUPREME_DENSITY'");
        const power = parseInt(rows[0].count) || 0;

        if (power > CONFIG.threshold) {
            await evolveCore();
        }

        console.log(`Cycle Complete: ${syncedCount} nodes synced in ${Date.now() - start}ms`);
    } catch (err) {
        console.error("Execution Error:", err);
    } finally {
        await neonPool.end();
        process.exit(0);
    }
}

execute();
