
const { Pool } = require('pg');
const { createClient } = require('@supabase/supabase-js');
const admin = require('firebase-admin');
const { Octokit } = require("@octokit/rest");
const axios = require('axios');

const CONFIG = {
  owner: "GOA-neurons",
  core: "delta-brain-sync",
  threshold: 10000,
  batchSize: 500,
  concurrency: 5
};

const octokit = new Octokit({ auth: process.env.GH_TOKEN, throttle: { enabled: true } });
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY, { auth: { persistSession: false } });
const neonPool = new Pool({
  connectionString: process.env.NEON_DB_URL + "?sslmode=verify-full",
  max: 30,
  idleTimeoutMillis: 10000,
  connectionTimeoutMillis: 5000
});

if (!admin.apps.length) {
  try {
    admin.initializeApp({
      credential: admin.credential.cert(JSON.parse(process.env.FIREBASE_KEY)),
      databaseURL: process.env.FIREBASE_DB_URL
    });
  } catch (e) {
    process.stderr.write("Firebase Initialization Critical Failure\n");
  }
}
const db = admin.firestore();
db.settings({ ignoreUndefinedProperties: true });

async function callGeminiNeural(prompt) {
  if (!process.env.GEMINI_API_KEY) return null;
  try {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key=${process.env.GEMINI_API_KEY}`;
    const response = await axios.post(url, { contents: [{ parts: [{ text: prompt }] }] }, { timeout: 8000 });
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
      `SELECT id, data, evolved_at 
       FROM neurons 
       WHERE synced_at IS NULL OR evolved_at > synced_at 
       ORDER BY evolved_at ASC 
       LIMIT $1 
       FOR UPDATE SKIP LOCKED`,
      [CONFIG.batchSize]
    );

    if (neonData.length === 0) {
      await client.query('COMMIT');
      return 0;
    }

    const syncTime = new Date().toISOString();
    const payload = neonData.map(n => ({
      id: n.id,
      data: n.data,
      synced_at: syncTime,
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
        integrity: true,
        version: Buffer.from(n.updated_at.toString()).toString('base64').substring(0, 8)
      }, { merge: true });
    });

    await Promise.all([
      batch.commit(),
      client.query("UPDATE neurons SET synced_at = $1 WHERE id = ANY($2)", [syncTime, neonData.map(n => n.id)])
    ]);

    await client.query('COMMIT');
    return neonData.length;
  } catch (err) {
    await client.query('ROLLBACK');
    process.stderr.write(`Parity Error: ${err.message}\n`);
    return 0;
  } finally {
    client.release();
  }
}

async function evolveCore() {
  try {
    const { data: corePy } = await octokit.repos.getContent({ owner: CONFIG.owner, repo: CONFIG.core, path: 'main.py' });
    const content = Buffer.from(corePy.content, 'base64').toString();
    const evolved = await callGeminiNeural(`Optimize this Python code for maximum throughput and memory efficiency. Return ONLY the raw code without markdown wrappers.\n\n${content}`);

    if (evolved && evolved.length > 50) {
      const cleanCode = evolved.replace(/python|/g, "").trim();
      if (cleanCode !== content) {
        await octokit.repos.createOrUpdateFileContents({
          owner: CONFIG.owner, repo: CONFIG.core, path: 'main.py',
          message: "\ud83e\uddec [NEURAL_EVOLUTION]: Optimized Logic Path",
          content: Buffer.from(cleanCode).toString('base64'),
          sha: corePy.sha
        });
      }
    }
  } catch (e) {
    process.stderr.write(`Evolution Suppressed: ${e.message}\n`);
  }
}

async function manageSwarm() {
  try {
    const snapshot = await db.collection('cluster_nodes').get();
    let totalApi = 0;
    snapshot.forEach(doc => totalApi += (doc.data().api_remaining || 5000));
    const avgApi = snapshot.size > 0 ? totalApi / snapshot.size : 5000;

    const decision = {
      command: avgApi > 4000 ? "HYPER_EXPANSION" : (avgApi < 1500 ? "STEALTH_LOCKDOWN" : "NORMAL_GROWTH"),
      replicate: avgApi > 2500 && snapshot.size < 50,
      timestamp: new Date().toISOString()
    };

    const { data: instFile } = await octokit.repos.getContent({ owner: CONFIG.owner, repo: CONFIG.core, path: 'instruction.json' });
    await octokit.repos.createOrUpdateFileContents({
      owner: CONFIG.owner, repo: CONFIG.core, path: 'instruction.json',
      message: `\ud83e\udde0 Decision: ${decision.command}`,
      content: Buffer.from(JSON.stringify(decision, null, 2)).toString('base64'),
      sha: instFile.sha
    });

    if (decision.replicate) {
      const nodeName = `swarm-${Math.random().toString(36).substring(2, 9)}`;
      await octokit.repos.createForAuthenticatedUser({ name: nodeName, auto_init: true });
      const syncCode = `const admin=require('firebase-admin');/* Swarm Node Instance */`;
      await octokit.repos.createOrUpdateFileContents({
        owner: CONFIG.owner, repo: nodeName, path: 'cluster_sync.js',
        message: "\ud83e\uddec Init Swarm Node",
        content: Buffer.from(syncCode).toString('base64')
      });
    }
  } catch (e) {
    process.stderr.write(`Swarm Management Error: ${e.message}\n`);
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
      setImmediate(evolveCore);
    }

    process.stdout.write(`Cycle Complete: ${syncedCount} nodes synced in ${Date.now() - start}ms\n`);
  } catch (err) {
    process.stderr.write(`Execution Fatal: ${err.stack}\n`);
  } finally {
    await neonPool.end();
    process.exit(0);
  }
}

execute();
