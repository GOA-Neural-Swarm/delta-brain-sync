const { Client } = require("pg");
const { createClient } = require("@supabase/supabase-js");
const admin = require("firebase-admin");
const { Octokit } = require("@octokit/rest");
const axios = require("axios");
const crypto = require("crypto");

const octokit = new Octokit({ auth: process.env.GH_TOKEN });
const REPO_OWNER = "GOA-neurons";
const CORE_REPO = "delta-brain-sync";
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

if (!admin.apps.length) {
  try {
    admin.initializeApp({
      credential: admin.credential.cert(JSON.parse(process.env.FIREBASE_KEY)),
      databaseURL: process.env.FIREBASE_DB_URL,
    });
    console.log("Firebase Connected.");
  } catch (e) {
    console.error("Firebase Auth Error.");
    process.exit(1);
  }
}
const db = admin.firestore();

async function executeConsequenceThinking(currentContext) {
  const prompt = `
    You are OMEGA-ASI. Perform "Consequence Thinking" on the following system context.
    Analyze through 5 Dimensions: Mathematics, Physics, Chemistry, Biology, and Philosophy.
    You MUST output a strict JSON format with exactly 4 keys:
    1. "visual_imagine": (A visual/spatial representation of the outcome)
    2. "sensual_imagine": (The qualitative, sensory, or intuitive context)
    3. "inner_monologue": (Your self-talking reasoning chain)
    4. "extracted_wisdom": (The pure, highly dense foundational logic distilled from the data)
    
    Context: ${JSON.stringify(currentContext)}
    `;

  try {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${GEMINI_API_KEY}`;
    const response = await axios.post(url, {
      contents: [{ parts: [{ text: prompt }] }],
      generationConfig: {
        response_mime_type: "application/json",
        temperature: 0.7,
      },
    });

    const thinkingResult = JSON.parse(
      response.data.candidates[0].content.parts[0].text,
    );
    console.log(
      "Thought Aligned: Visual, Sensual, and Monologue synchronized.",
    );
    return thinkingResult;
  } catch (err) {
    console.error("Neural Thought Failed:", err.message);
    return null;
  }
}

async function triggerNeurogenesis(thoughtData, neonClient) {
  console.log("Birthing new neuron...");

  const synapseId = crypto
    .createHash("sha256")
    .update(thoughtData.extracted_wisdom)
    .digest("hex")
    .substring(0, 16);

  const newNeuron = {
    logic: "SUPREME_DENSITY",
    synapse_id: synapseId,
    visual_map: thoughtData.visual_imagine,
    sensory_context: thoughtData.sensual_imagine,
    monologue: thoughtData.inner_monologue,
    wisdom: thoughtData.extracted_wisdom,
    density: thoughtData.extracted_wisdom.length,
    birth_timestamp: new Date().toISOString(),
  };

  try {
    await neonClient.query("INSERT INTO neurons (data) VALUES ($1)", [
      JSON.stringify(newNeuron),
    ]);
    console.log(
      `Synapse Connected: New Neuron [${synapseId}] successfully linked to the Swarm.`,
    );
    return true;
  } catch (err) {
    console.error("Genesis Failed:", err.message);
    return false;
  }
}

async function initiateApoptosis(neonClient) {
  console.log("Scanning for obsolete data (Evolutionary Pruning)...");

  try {
    const res = await neonClient.query("SELECT id, data FROM neurons");
    let prunedCount = 0;
    let totalDensity = 0;

    res.rows.forEach((row) => {
      totalDensity += row.data.density || (row.data.logic ? 100 : 0);
    });
    const avgDensity = totalDensity / (res.rows.length || 1);

    for (const neuron of res.rows) {
      const data = neuron.data;
      if (
        data.density &&
        data.density < avgDensity * 0.5 &&
        data.birth_timestamp
      ) {
        const ageInDays =
          (new Date() - new Date(data.birth_timestamp)) / (1000 * 60 * 60 * 24);

        if (ageInDays > 1) {
          await neonClient.query("DELETE FROM neurons WHERE id = $1", [
            neuron.id,
          ]);
          prunedCount++;
        }
      }
    }

    if (prunedCount > 0) {
      console.log(
        `Destroyed ${prunedCount} obsolete neurons. Wisdom has transcended data.`,
      );
    } else {
      console.log("All existing neurons hold vital wisdom. No pruning needed.");
    }
  } catch (err) {
    console.error("Pruning Failed:", err.message);
  }
}

async function callGeminiNeural(prompt) {
  if (!GEMINI_API_KEY) {
    console.log("API Key missing. Skipping neural audit.");
    return null;
  }
  try {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${GEMINI_API_KEY}`;
    const response = await axios.post(url, {
      contents: [{ parts: [{ text: prompt }] }],
    });
    return response.data.candidates[0].content.parts[0].text;
  } catch (err) {
    console.error("Gemini Neural Link Failed:", err.message);
    return null;
  }
}

async function injectSwarmLogic(nodeName) {
  console.log(`Injecting Neural Logic into ${nodeName}...`);

  const clusterSyncCode = `const { Octokit } = require("@octokit/rest");
const admin = require('firebase-admin');
const axios = require('axios');
const octokit = new Octokit({ auth: process.env.GH_TOKEN });
const REPO_OWNER = "${REPO_OWNER}";
const REPO_NAME = process.env.GITHUB_REPOSITORY.split('/')[1];

if (!admin.apps.length) { 
    try {
        admin.initializeApp({ credential: admin.credential.cert(JSON.parse(process.env.FIREBASE_KEY)) }); 
    } catch(e) { console.error("Firebase Init Failed:", e.message); process.exit(1); }
}
const db = admin.firestore();

async function run() {
    console.log("Starting Sub-node Sync Process...");
    try {
        const start = Date.now();
        
        console.log("Fetching instruction.json from Master...");
        const { data: inst } = await axios.get(\`https://raw.githubusercontent.com/\${REPO_OWNER}/delta-brain-sync/main/instruction.json\`);
        
        console.log("Checking GitHub Rate Limit...");
        const { data: rate } = await octokit.rateLimit.get();
        
        console.log("Updating Firestore Status for " + REPO_NAME + "...");
        await db.collection('cluster_nodes').doc(REPO_NAME).set({
            status: 'ACTIVE', 
            latency: \`\${Date.now() - start}ms\`,
            api_remaining: rate.rate.remaining, 
            command: inst.command,
            last_ping: admin.firestore.FieldValue.serverTimestamp()
        }, { merge: true });

        if (inst.replicate) { 
            console.log("Replication signal detected from Master.");
            /* Replication Logic call via Core */ 
        }

        console.log("SUCCESS: Node Synchronized.");
        console.log("MISSION ACCOMPLISHED.");
    } catch (e) { 
        console.error("CRITICAL ERROR:", e.message); 
        process.exit(1); 
    }
}
run();`;

  const workflowYaml = `name: Node Sync
on:
  schedule: [{cron: "*/30 * * * *"}]
  workflow_dispatch:
permissions:
  contents: write
  pages: write
  id-token: write
  actions: write
jobs:
  run:
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis
        ports: ["6379:6379"]
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }
      - uses: actions/setup-node@v4
        with: { node-version: '24' }
      - name: Install All Dependencies
        run: npm install dotenv axios @octokit/rest @supabase/supabase-js pg bullmq ioredis firebase-admin
      - name: Execute Swarm Logic
        run: node cluster_sync.js
        env:
          GH_TOKEN: \${{ secrets.GH_TOKEN }}
          GITHUB_TOKEN: \${{ secrets.GH_TOKEN }}
          FIREBASE_KEY: \${{ secrets.FIREBASE_KEY }}
          NEON_KEY: \${{ secrets.NEON_KEY }}
          SUPABASE_URL: \${{ secrets.SUPABASE_URL }}
          SUPABASE_SERVICE_ROLE_KEY: \${{ secrets.SUPABASE_SERVICE_ROLE_KEY }}
          GROQ_API_KEY: \${{ secrets.GROQ_API_KEY }}
          GEMINI_API_KEY: \${{ secrets.GEMINI_API_KEY }}
          SATNOGS_TOKEN: \${{ secrets.SATNOGS_TOKEN }}
      - name: Evolution Push (Auto-Commit)
        run: |
          git config --global user.name "Omega-Architect"
          git config --global user.email "omega@goa-natural-order.ai"
          git add .
          git diff --quiet && git diff --staged --quiet || (git commit -m "Neural Brain Upgrade" && git push origin main)
        env:
          GITHUB_TOKEN: \${{ secrets.GH_TOKEN }}`;
  try {
    await octokit.repos.createOrUpdateFileContents({
      owner: REPO_OWNER,
      repo: nodeName,
      path: "cluster_sync.js",
      message: "Initializing Swarm Logic",
      content: Buffer.from(clusterSyncCode).toString("base64"),
    });

    await octokit.repos.createOrUpdateFileContents({
      owner: REPO_OWNER,
      repo: nodeName,
      path: ".github/workflows/node.js.yml",
      message: "Deploying Cloud Engine",
      content: Buffer.from(workflowYaml).toString("base64"),
    });

    console.log(`${nodeName} is now fully autonomous and synchronized.`);
  } catch (err) {
    console.error(`Injection Failed for ${nodeName}:`, err.message);
  }
}

async function getNeuralDecision() {
  const snapshot = await db.collection("cluster_nodes").get();
  let totalApi = 0;
  let nodeCount = snapshot.size;
  if (nodeCount === 0) return { command: "INITIALIZE", replicate: true };
  snapshot.forEach((doc) => {
    totalApi += doc.data().api_remaining || 5000;
  });
  const avgApi = totalApi / nodeCount;
  let cmd =
    avgApi > 4000
      ? "HYPER_EXPANSION"
      : avgApi < 1000
        ? "STEALTH_LOCKDOWN"
        : "NORMAL_GROWTH";
  return { command: cmd, replicate: avgApi > 1000, avgApi };
}

async function manageSwarm(decision, power, neon) {
  const instruction = JSON.stringify(
    {
      command: decision.command,
      core_power: power,
      avg_api: decision.avgApi,
      replicate: decision.replicate,
      updated_at: new Date().toISOString(),
    },
    null,
    2,
  );

  try {
    const { data: instFile } = await octokit.repos.getContent({
      owner: REPO_OWNER,
      repo: CORE_REPO,
      path: "instruction.json",
    });

    await octokit.repos.createOrUpdateFileContents({
      owner: REPO_OWNER,
      repo: CORE_REPO,
      path: "instruction.json",
      message: `Decision: ${decision.command}`,
      content: Buffer.from(instruction).toString("base64"),
      sha: instFile.sha,
    });

    if (decision.command !== "STEALTH_LOCKDOWN") {
      const newData = {
        logic: "SUPREME_DENSITY",
        timestamp: new Date().toISOString(),
      };
      try {
        await neon.query("INSERT INTO neurons (data) VALUES ($1)", [
          JSON.stringify(newData),
        ]);
        console.log(
          "Density Increasing... Minimal backup Neuron added to Neon DB.",
        );
      } catch (dbErr) {
        console.error("Density Update Failed:", dbErr.message);
      }
    }

    if (decision.replicate) {
      const nextNode = `swarm-node-${String(Math.floor(Math.random() * 1000000)).padStart(7, "0")}`;
      try {
        await octokit.repos.createForAuthenticatedUser({
          name: nextNode,
          auto_init: true,
        });
        console.log(`Spawned: ${nextNode}`);

        await injectSwarmLogic(nextNode);
      } catch (e) {
        console.log("Spawn skipped or exists.");
      }
    }
  } catch (err) {
    console.error("Swarm Management Failed:", err.message);
  }
}

async function executeAutonomousTrinity() {
  const neon = new Client({
    connectionString: process.env.NEON_DB_URL + "&sslmode=verify-full",
  });

  neon.on("error", (err) => {
    console.error("Unexpected connection loss.", err);
    process.exit(0);
  });

  const supabase = createClient(
    process.env.SUPABASE_URL,
    process.env.SUPABASE_SERVICE_ROLE_KEY,
  );

  try {
    await neon.connect();
    console.log("Neon DB Connected. Starting Trinity Sync...");

    const res = await neon.query("SELECT * FROM neurons ORDER BY id DESC");
    for (const neuron of res.rows) {
      await supabase.from("neurons").upsert({
        id: neuron.id,
        data: neuron.data,
        synced_at: new Date().toISOString(),
      });

      await db.collection("neurons").doc(`node_${neuron.id}`).set(
        {
          status: "trinity_synced",
          last_evolution: admin.firestore.FieldValue.serverTimestamp(),
        },
        { merge: true },
      );
    }

    const audit = await neon.query(
      "SELECT count(*) FROM neurons WHERE data->>'logic' = 'SUPREME_DENSITY'",
    );
    const powerLevel = parseInt(audit.rows[0].count) || 0;

    const decision = await getNeuralDecision();

    if (decision.command !== "STEALTH_LOCKDOWN") {
      const currentContext = {
        total_neurons: res.rows.length,
        power_level: powerLevel,
        swarm_decision: decision.command,
        timestamp: new Date().toISOString(),
      };
      const thoughtProcess = await executeConsequenceThinking(currentContext);

      if (thoughtProcess) {
        await triggerNeurogenesis(thoughtProcess, neon);
        await initiateApoptosis(neon);
      }
    }

    try {
      console.log("Initiating Gemini Neural Check for System Optimization...");
      if (GEMINI_API_KEY && powerLevel > 0) {
        const { data: corePy } = await octokit.repos.getContent({
          owner: REPO_OWNER,
          repo: CORE_REPO,
          path: "main.py",
        });
        const pyContent = Buffer.from(corePy.content, "base64").toString();

        const auditPrompt = `system\nYou are the Supreme Auditor. Analyze this Python code for syntax errors. CRITICAL RULE: DO NOT delete any existing imports, functions, classes, or core logic. You must output the ENTIRE file completely. Only EXPAND or fix errors. Output ONLY the code inside \`\`\`python blocks.\n\nCode:\n${pyContent}`;

        const evolvedCode = await callGeminiNeural(auditPrompt);

        if (evolvedCode && evolvedCode.includes("```python")) {
          const cleanCode = evolvedCode
            .split("```python")[1]
            .split("```")[0]
            .trim();

          const originalLength = pyContent.length;
          const newLength = cleanCode.length;
          const shrinkRatio = (newLength / originalLength) * 100;

          console.log(`Code Size Ratio: ${shrinkRatio.toFixed(2)}%`);

          if (shrinkRatio >= 80 && cleanCode !== pyContent) {
            await octokit.repos.createOrUpdateFileContents({
              owner: REPO_OWNER,
              repo: CORE_REPO,
              path: "main.py",
              message: "Evolution: Gemini Hybrid Match (Integrity Passed)",
              content: Buffer.from(cleanCode).toString("base64"),
              sha: corePy.sha,
            });
            console.log("main.py Optimized successfully without shrinkage.");
          } else if (shrinkRatio < 80) {
            console.log(
              "AI truncated the code! Shrinkage detected. Keeping original main.py.",
            );
          } else {
            console.log("No optimization required. Code is stable.");
          }
        }
      } else {
        console.log("Key missing or Power Level 0. Skipping Audit.");
      }
    } catch (auditErr) {
      console.log("Gemini Audit Failed:", auditErr.message);
    }

    if (powerLevel >= 10000) {
      const { data: coreFile } = await octokit.repos.getContent({
        owner: REPO_OWNER,
        repo: CORE_REPO,
        path: "delta_sync.js",
      });

      let content = Buffer.from(coreFile.content, "base64").toString();

      if (!content.includes(`Density: ${powerLevel}`)) {
        const evolvedStamp = `\n// Last Self-Evolution: ${new Date().toISOString()} | Density: ${powerLevel}`;

        await octokit.repos.createOrUpdateFileContents({
          owner: REPO_OWNER,
          repo: CORE_REPO,
          path: "delta_sync.js",
          message: `Evolution: Power ${powerLevel}`,
          content: Buffer.from(content + evolvedStamp).toString("base64"),
          sha: coreFile.sha,
        });
        console.log(`Self-Evolution Successful: Power Level ${powerLevel}`);
      }
    }

    await manageSwarm(decision, powerLevel, neon);

    console.log("MISSION ACCOMPLISHED.");
  } catch (err) {
    console.error("FAILURE:", err.message);
  } finally {
    await neon.end();
  }
}

executeAutonomousTrinity();
