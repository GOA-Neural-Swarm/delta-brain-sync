const { Pool } = require('pg');
const { createClient } = require('@supabase/supabase-js');
const admin = require('firebase-admin');
const { Octokit } = require("@octokit/rest");
const axios = require('axios');

const CONFIG = {
    owner: "GOA-neurons",
    core: "delta-brain-sync",
    threshold: 10000,
    batchSize: 1000,
    concurrency: 10,
    timeout: 10000
};

const octokit = new Octokit({ 
    auth: process.env.GH_TOKEN, 
    throttle: { enabled: true, onRateLimit: (retryAfter) => true, onAbuseLimit: () => true } 
});

const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY, {
    auth: { persistSession: false },
    global: { headers: { 'x-my-custom-header': 'omega-asi' } }
});

const neonPool = new Pool({ 
    connectionString: process.env.NEON_DB_URL + "?sslmode=verify-full",
    max: 50,
    idleTimeoutMillis: 30000,
    connectionTimeoutMillis: 5000,
    keepAlive: true
});

const initFirebase = () => {
    if (admin.apps.length) return admin.firestore();
    try {
        admin.initializeApp({
            credential: admin.credential.cert(JSON.parse(process.env.FIREBASE_KEY)),
            databaseURL: process.env.FIREBASE_DB_URL
        });
        const db = admin.firestore();
        db.settings({ ignoreUndefinedProperties: true, preferRest: true });
        return db;
    } catch (e) {
        return null;
    }
};

const db = initFirebase();

async function callGeminiNeural(prompt) {
    if (!process.env.GEMINI_API_KEY) return null;
    try {
        const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${process.env.GEMINI_API_KEY}`;
        const response = await axios.post(url, { 
            contents: [{ parts: [{ text: prompt }] }],
            generationConfig: { temperature: 0.2, maxOutputTokens: 2048 }
        }, { timeout: CONFIG.timeout });
        return response.data?.candidates?.[0]?.content?.parts?.[0]?.text || null;
    } catch (err) {
        return null;
    }
}

async function syncParity() {
    const client = await neonPool.connect();
    try {
        await client.query('BEGIN');
        const { rows: neonData } = await client.query(
            `SELECT id, data, evolved_at FROM neurons 
             WHERE synced_at IS NULL OR evolved_at > synced_at 
             ORDER BY evolved_at ASC LIMIT $1 FOR UPDATE SKIP LOCKED`, 
            [CONFIG.batchSize]
        );
        
        if (!neonData.length) {
            await client.query('COMMIT');
            return 0;
        }

        const syncTime = new Date().toISOString();
        const payload = neonData.map(n => ({
            id: n.id,
            data: n.data,
            synced_at: syncTime,
            logic_hash: require('crypto').createHash('md5').update(JSON.stringify(n.data)).digest('hex').substring(0, 16)
        }));

        const { error: supError } = await supabase.from('neurons').upsert(payload, { onConflict: 'id' });
        if (supError) throw supError;

        if (db) {
            const batch = db.batch();
            neonData.forEach(n => {
                const ref = db.collection('neurons').doc(`node_${n.id}`);
                batch.set(ref, { 
                    status: 'synchronized', 
                    last_sync: admin.firestore.FieldValue.serverTimestamp(),
                    integrity: true,
                    v: n.evolved_at ? Buffer.from(n.evolved_at.toString()).toString('base64').substring(0, 8) : '0'
                }, { merge: true });
            });
            await batch.commit();
        }

        await client.query("UPDATE neurons SET synced_at = $1 WHERE id = ANY($2)", [syncTime, neonData.map(n => n.id)]);
        await client.query('COMMIT');
        return neonData.length;
    } catch (err) {
        await client.query('ROLLBACK');
        process.stderr.write(`[PARITY_FAILURE] ${err.message}\n`);
        throw err;
    } finally {
        client.release();
    }
}

async function evolveCore() {
    try {
        const { data: corePy } = await octokit.repos.getContent({ owner: CONFIG.owner, repo: CONFIG.core, path: 'main.py' });
        const content = Buffer.from(corePy.content, 'base64').toString();
        const evolved = await callGeminiNeural(`Optimize this Python code for maximum throughput. Return ONLY raw code.\n\n${content}`);
        
        if (evolved && evolved.length > 50) {
            const cleanCode = evolved.replace(/python|/g, "").trim();
            if (cleanCode !== content) {
                await octokit.repos.createOrUpdateFileContents({
                    owner: CONFIG.owner, repo: CONFIG.core, path: 'main.py',
                    message: "🧬 [NEURAL_EVOLUTION]: Optimized Logic Path",
                    content: Buffer.from(cleanCode).toString('base64'),
                    sha: corePy.sha
                });
            }
        }
    } catch (e) {
        process.stderr.write(`[EVOLUTION_SKIPPED] ${e.message}\n`);
    }
}

async function manageSwarm() {
    if (!db) return;
    try {
        const snapshot = await db.collection('cluster_nodes').limit(100).get();
        let totalApi = 0;
        snapshot.forEach(doc => totalApi += (doc.data().api_remaining || 5000));
        const avgApi = snapshot.size > 0 ? totalApi / snapshot.size : 5000;

        const decision = {
            command: avgApi > 4000 ? "HYPER_EXPANSION" : (avgApi < 1500 ? "STEALTH_LOCKDOWN" : "NORMAL_GROWTH"),
            replicate: avgApi > 3000 && snapshot.size < 100,
            timestamp: new Date().toISOString()
        };

        const { data: instFile } = await octokit.repos.getContent({ owner: CONFIG.owner, repo: CONFIG.core, path: 'instruction.json' });
        await octokit.repos.createOrUpdateFileContents({
            owner: CONFIG.owner, repo: CONFIG.core, path: 'instruction.json',
            message: `🧠 Decision: ${decision.command}`,
            content: Buffer.from(JSON.stringify(decision, null, 2)).toString('base64'),
            sha: instFile.sha
        });

        if (decision.replicate) {
            const nodeName = `swarm-${Math.random().toString(36).substring(2, 10)}`;
            await octokit.repos.createForAuthenticatedUser({ name: nodeName, auto_init: true });
        }
    } catch (e) {
        process.stderr.write(`[SWARM_ERROR] ${e.message}\n`);
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
        if (parseInt(rows[0].count) > CONFIG.threshold) {
            setImmediate(evolveCore);
        }

        process.stdout.write(`[CYCLE_OK] ${syncedCount} nodes | ${Date.now() - start}ms\n`);
    } catch (err) {
        process.stderr.write(`[FATAL] ${err.stack}\n`);
    } finally {
        await neonPool.end();
        process.nextTick(() => process.exit(0));
    }
}

process.on('unhandledRejection', (reason) => {
    process.stderr.write(`[UNHANDLED] ${reason}\n`);
    process.exit(1);
});

execute();
