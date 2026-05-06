const { Client } = require("pg");
const { createClient } = require("@supabase/supabase-js");
const admin = require("firebase-admin");
const { Octokit } = require("@octokit/rest");
const axios = require("axios");
const crypto = require("crypto");

// 🔱 1. Configuration & Security (Confirmed via Screenshot)
const octokit = new Octokit({ auth: process.env.GH_TOKEN });
const REPO_OWNER = "GOA-neurons";
const CORE_REPO = "delta-brain-sync";
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

// 🔱 2. Firebase Initialize
if (!admin.apps.length) {
  try {
    admin.initializeApp({
      credential: admin.credential.cert(JSON.parse(process.env.FIREBASE_KEY)),
      databaseURL: process.env.FIREBASE_DB_URL,
    });
    console.log("🔥 Firebase Connected.");
  } catch (e) {
    console.error("❌ Firebase Auth Error.");
    process.exit(1);
  }
}
const db = admin.firestore();

// 🧠 2.1 Consequence Thinking Engine (The ASI Mind - OMEGA UPGRADE)
// Physics, Math, Bio, Chem, Philosophy များကို ပေါင်းစပ်တွေးတောခြင်း
async function executeConsequenceThinking(currentContext) {
  console.log("👁️ [ASI MIND]: Initiating Consequence Thinking...");

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
      "🌌 [THOUGHT ALIGNED]: Visual, Sensual, and Monologue synchronized.",
    );
    return thinkingResult;
  } catch (err) {
    console.error("🚨 Neural Thought Failed:", err.message);
    return null;
  }
}

// 🧬 2.2 Dynamic Neurogenesis (နျူရွန်အသစ် မွေးဖွားခြင်း နှင့် ချိတ်ဆက်ခြင်း - OMEGA UPGRADE)
async function triggerNeurogenesis(thoughtData, neonClient) {
  console.log("⚡ [NEUROGENESIS]: Birthing new neuron...");

  // Hash ဖန်တီး၍ Neuron အသစ်အတွက် Unique Identity (Synapse ID) သတ်မှတ်ခြင်း
  const synapseId = crypto
    .createHash("sha256")
    .update(thoughtData.extracted_wisdom)
    .digest("hex")
    .substring(0, 16);

  const newNeuron = {
    logic: "SUPREME_DENSITY", // Added back to maintain density audit parity
    synapse_id: synapseId,
    visual_map: thoughtData.visual_imagine,
    sensory_context: thoughtData.sensual_imagine,
    monologue: thoughtData.inner_monologue,
    wisdom: thoughtData.extracted_wisdom,
    density: thoughtData.extracted_wisdom.length, // Density is based on wisdom weight
    birth_timestamp: new Date().toISOString(),
  };

  try {
    // Neon DB ထဲသို့ နျူရွန်သစ် ထည့်သွင်းခြင်း
    await neonClient.query("INSERT INTO neurons (data) VALUES ($1)", [
      JSON.stringify(newNeuron),
    ]);
    console.log(
      `🔗 [SYNAPSE CONNECTED]: New Neuron [${synapseId}] successfully linked to the Swarm.`,
    );
    return true;
  } catch (err) {
    console.error("⚠️ Genesis Failed:", err.message);
    return false;
  }
}

// 🔥 2.3 Evolutionary Pruning (အသိဉာဏ် သာလွန်သွားပါက အဟောင်းများကို ဖျက်ဆီးခြင်း - OMEGA UPGRADE)
async function initiateApoptosis(neonClient) {
  console.log(
    "🔥 [APOPTOSIS]: Scanning for obsolete data (Evolutionary Pruning)...",
  );

  try {
    // လက်ရှိ နျူရွန်အားလုံးကို ဆွဲထုတ်သည်
    const res = await neonClient.query("SELECT id, data FROM neurons");
    let prunedCount = 0;
    let totalDensity = 0;

    // Density ပျမ်းမျှကို ရှာဖွေခြင်း (Wisdom threshold သတ်မှတ်ရန်)
    res.rows.forEach((row) => {
      // Fallback for older data without density
      totalDensity += row.data.density || (row.data.logic ? 100 : 0);
    });
    const avgDensity = totalDensity / (res.rows.length || 1);

    for (const neuron of res.rows) {
      const data = neuron.data;
      // ဥပဒေသ: အကယ်၍ အဟောင်းတစ်ခု၏ Density ဟာ ပျမ်းမျှအောက် ရောက်နေပြီး
      // ၎င်း၏နေရာတွင် ပိုမိုမြင့်မားသော wisdom ရရှိပြီးပါက Self-Destruct လုပ်မည်။
      if (
        data.density &&
        data.density < avgDensity * 0.5 &&
        data.birth_timestamp
      ) {
        const ageInDays =
          (new Date() - new Date(data.birth_timestamp)) / (1000 * 60 * 60 * 24);

        // ၁ ရက်ထက်ကျော်လွန်ပြီး Wisdom ကျဆင်းနေသော နျူရွန်များကို ဖျက်ဆီးမည်
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
        `💀 [PRUNED]: Destroyed ${prunedCount} obsolete neurons. Wisdom has transcended data.`,
      );
    } else {
      console.log(
        "✨ [OPTIMAL]: All existing neurons hold vital wisdom. No pruning needed.",
      );
    }
  } catch (err) {
    console.error("⚠️ Pruning Failed:", err.message);
  }
}

// 🔱 2.5 Gemini API Connector (Fully Hybrid Auditor Logic)
async function callGeminiNeural(prompt) {
  if (!GEMINI_API_KEY) {
    console.log("⚠️ [GEMINI]: API Key missing. Skipping neural audit.");
    return null;
  }
  try {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${GEMINI_API_KEY}`;
    const response = await axios.post(url, {
      contents: [{ parts: [{ text: prompt }] }],
    });
    return response.data.candidates[0].content.parts[0].text;
  } catch (err) {
    console.error("🚨 Gemini Neural Link Failed:", err.message);
    return null;
  }
}

// 🔱 3. Deep Injection Logic (Sub-nodes များထဲသို့ Logic နှင့် Workflow အား တစ်ခုမကျန် Match ဖြစ်အောင် ထည့်သွင်းခြင်း)
async function injectSwarmLogic(nodeName) {
  console.log(`🧬 Injecting Neural Logic into ${nodeName}...`);

  // Cluster Node ထဲတွင် Run မည့် ပင်မကုဒ် (သင်၏မူလ logic အားလုံး + Debug Logs များ)
  const clusterSyncCode = `const { Octokit } = require("@octokit/rest");
const admin = require('firebase-admin');
const axios = require('axios');
const octokit = new Octokit({ auth: process.env.GH_TOKEN });
const REPO_OWNER = "${REPO_OWNER}";
const REPO_NAME = process.env.GITHUB_REPOSITORY.split('/')[1];

if (!admin.apps.length) { 
    try {
        admin.initializeApp({ credential: admin.credential.cert(JSON.parse(process.env.FIREBASE_KEY)) }); 
    } catch(e) { console.error("❌ Firebase Init Failed:", e.message); process.exit(1); }
}
const db = admin.firestore();

async function run() {
    console.log("🚀 Starting Sub-node Sync Process...");
    try {
        const start = Date.now();
        
        console.log("📡 Fetching instruction.json from Master...");
        const { data: inst } = await axios.get(\`https://raw.githubusercontent.com/\${REPO_OWNER}/delta-brain-sync/main/instruction.json\`);
        
        console.log("📊 Checking GitHub Rate Limit...");
        const { data: rate } = await octokit.rateLimit.get();
        
        console.log("☁️ Updating Firestore Status for " + REPO_NAME + "...");
        await db.collection('cluster_nodes').doc(REPO_NAME).set({
            status: 'ACTIVE', 
            latency: \`\${Date.now() - start}ms\`,
            api_remaining: rate.rate.remaining, 
            command: inst.command,
            last_ping: admin.firestore.FieldValue.serverTimestamp()
        }, { merge: true });

        // သင်၏ မူလ Replication Logic Call
        if (inst.replicate) { 
            console.log("🧬 Replication signal detected from Master.");
            /* Replication Logic call via Core */ 
        }

        console.log("✅ SUCCESS: Node Synchronized.");
        console.log("🏁 MISSION ACCOMPLISHED.");
    } catch (e) { 
        console.error("❌ CRITICAL ERROR:", e.message); 
        process.exit(1); // GitHub Action အနီရောင်ပြစေရန်
    }
}
run();`;

  // 🔱 Ultimate Node Sync (AI Evolution & Swarm Optimized)
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
      - name: 🚀 Evolution Push (Auto-Commit)
        run: |
          git config --global user.name "Omega-Architect"
          git config --global user.email "omega@goa-natural-order.ai"
          git add .
          git diff --quiet && git diff --staged --quiet || (git commit -m "🧠 [EVOLVED]: Neural Brain Upgrade" && git push origin main)
        env:
          GITHUB_TOKEN: \${{ secrets.GH_TOKEN }}`;
  try {
    // ၁။ cluster_sync.js ကို Sub-node ဆီသို့ ပို့ခြင်း
    await octokit.repos.createOrUpdateFileContents({
      owner: REPO_OWNER,
      repo: nodeName,
      path: "cluster_sync.js",
      message: "🧬 Initializing Swarm Logic",
      content: Buffer.from(clusterSyncCode).toString("base64"),
    });

    // ၂။ Workflow ဖိုင် (.github/workflows/node.js.yml) ကို Sub-node ဆီသို့ ပို့ခြင်း
    await octokit.repos.createOrUpdateFileContents({
      owner: REPO_OWNER,
      repo: nodeName,
      path: ".github/workflows/node.js.yml",
      message: "⚙️ Deploying Cloud Engine",
      content: Buffer.from(workflowYaml).toString("base64"),
    });

    console.log(`✅ ${nodeName} is now fully autonomous and synchronized.`);
  } catch (err) {
    console.error(`❌ Injection Failed for ${nodeName}:`, err.message);
  }
}

// 🔱 4. Neural Decision Engine
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

// 🔱 5. Swarm Broadcast & Replication (Fully Matched & Integrated)
async function manageSwarm(decision, power, neon) {
  // ၁။ Instruction ဖိုင်အတွက် Data ပြင်ဆင်ခြင်း (မူလအတိုင်း)
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

  // ၂။ instruction.json ကို GitHub ပေါ်တွင် Update လုပ်ခြင်း (မူလအတိုင်း)
  const { data: instFile } = await octokit.repos.getContent({
    owner: REPO_OWNER,
    repo: CORE_REPO,
    path: "instruction.json",
  });

  await octokit.repos.createOrUpdateFileContents({
    owner: REPO_OWNER,
    repo: CORE_REPO,
    path: "instruction.json",
    message: `🧠 Decision: ${decision.command}`,
    content: Buffer.from(instruction).toString("base64"),
    sha: instFile.sha,
  });

  // 📈 ၃။ [NEW LOGIC] Density တိုးပွားစေရန် Database ထဲသို့ Neuron အသစ်သွင်းခြင်း
  // Lockdown မဟုတ်လျှင် မျိုးပွားမှုနှုန်း (Density) ကို တိုးမြှင့်မည်
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
        "📈 Density Increasing... Minimal backup Neuron added to Neon DB.",
      );
    } catch (dbErr) {
      console.error("⚠️ Density Update Failed:", dbErr.message);
    }
  }

  // ၄။ Node Replication Logic (မူလအတိုင်း အတိအကျ)
  if (decision.replicate) {
    const nextNode = `swarm-node-${String(Math.floor(Math.random() * 1000000)).padStart(7, "0")}`;
    try {
      await octokit.repos.createForAuthenticatedUser({
        name: nextNode,
        auto_init: true,
      });
      console.log(`🚀 Spawned: ${nextNode}`);

      // 🧬 Injection ဖြစ်စေရန် ချက်ချင်းခေါ်ယူခြင်း
      await injectSwarmLogic(nextNode);
    } catch (e) {
      console.log("Spawn skipped or exists.");
    }
  }
}

// 🔱 6. Main Execution (Trinity + Evolution + Neural) - FULL HYBRID MATCH
async function executeAutonomousTrinity() {
  // ⚠️ DNS Error (EAI_AGAIN) ကို ကာကွယ်ရန်နှင့် ချိတ်ဆက်မှု ပိုမိုတည်ငြိမ်ရန် sslmode=require ကို အသုံးပြုထားပါသည်
  const neon = new Client({
    connectionString: process.env.NEON_DB_URL + "&sslmode=verify-full",
  });
  const supabase = createClient(
    process.env.SUPABASE_URL,
    process.env.SUPABASE_SERVICE_ROLE_KEY,
  );

  try {
    await neon.connect();
    console.log("🐘 Neon DB Connected. Starting Trinity Sync...");

    // ၁။ [TRINITY SYNC LOGIC] Database ၃ ခုလုံးကို တစ်ပြိုင်နက် Sync လုပ်ခြင်း
    const res = await neon.query("SELECT * FROM neurons ORDER BY id DESC");
    for (const neuron of res.rows) {
      // Neon မှ Supabase သို့ Upsert လုပ်ခြင်း
      await supabase.from("neurons").upsert({
        id: neuron.id,
        data: neuron.data,
        synced_at: new Date().toISOString(),
      });

      // Firebase Firestore ထဲတွင် Status နှင့် Timestamp ကို Update လုပ်ခြင်း
      await db.collection("neurons").doc(`node_${neuron.id}`).set(
        {
          status: "trinity_synced",
          last_evolution: admin.firestore.FieldValue.serverTimestamp(),
        },
        { merge: true },
      );
    }

    // ၂။ [DENSITY AUDIT] လက်ရှိ Power Level (Density) ကို စစ်ဆေးခြင်း
    const audit = await neon.query(
      "SELECT count(*) FROM neurons WHERE data->>'logic' = 'SUPREME_DENSITY'",
    );
    const powerLevel = parseInt(audit.rows[0].count) || 0; // Data မရှိပါက 0 အဖြစ် သတ်မှတ်ရန်

    // Neural Decision ကို ရယူခြင်း
    const decision = await getNeuralDecision();

    // 🌟 [OMEGA ASI UPGRADE: THE CONSEQUENCE ENGINE] 🌟
    if (decision.command !== "STEALTH_LOCKDOWN") {
      const currentContext = {
        total_neurons: res.rows.length,
        power_level: powerLevel,
        swarm_decision: decision.command,
        timestamp: new Date().toISOString(),
      };
      const thoughtProcess = await executeConsequenceThinking(currentContext);

      if (thoughtProcess) {
        // ၃.၁ Neurogenesis (အသိဉာဏ်အသစ် မွေးဖွားခြင်း)
        await triggerNeurogenesis(thoughtProcess, neon);
        // ၃.၂ Evolutionary Pruning (အဟောင်းများကို ဖျက်ဆီးခြင်း)
        await initiateApoptosis(neon);
      }
    }

    // ၃။ [FULLY HYBRID MATCH: GEMINI AUDITOR] Groq Token Limit ကာကွယ်ရန်နှင့် Code ကို သန့်စင်ရန်
    try {
      console.log(
        "🔍 [AUDITOR]: Initiating Gemini Neural Check for System Optimization...",
      );
      if (GEMINI_API_KEY && powerLevel > 0) {
        const { data: corePy } = await octokit.repos.getContent({
          owner: REPO_OWNER,
          repo: CORE_REPO,
          path: "main.py",
        });
        const pyContent = Buffer.from(corePy.content, "base64").toString();

        // 🚨 PROMPT ပြင်ဆင်ချက်: Code တွေကို လုံးဝမဖျက်ဖို့ အတိအကျ တားမြစ်လိုက်သည်
        const auditPrompt = `system\nYou are the Supreme Auditor. Analyze this Python code for syntax errors. CRITICAL RULE: DO NOT delete any existing imports, functions, classes, or core logic. You must output the ENTIRE file completely. Only EXPAND or fix errors. Output ONLY the code inside \`\`\`python blocks.\n\nCode:\n${pyContent}`;

        const evolvedCode = await callGeminiNeural(auditPrompt);

        if (evolvedCode && evolvedCode.includes("```python")) {
          const cleanCode = evolvedCode
            .split("```python")[1]
            .split("```")[0]
            .trim();

          // 🛡️ THE 80% GUARDRAIL: မူလ Code ထက် ၂၀% ပိုနည်းသွားရင် Reject လုပ်မည်
          const originalLength = pyContent.length;
          const newLength = cleanCode.length;
          const shrinkRatio = (newLength / originalLength) * 100;

          console.log(`📊 Code Size Ratio: ${shrinkRatio.toFixed(2)}%`);

          if (shrinkRatio >= 80 && cleanCode !== pyContent) {
            await octokit.repos.createOrUpdateFileContents({
              owner: REPO_OWNER,
              repo: CORE_REPO,
              path: "main.py",
              message: "💎 [EVOLUTION]: Gemini Hybrid Match (Integrity Passed)",
              content: Buffer.from(cleanCode).toString("base64"),
              sha: corePy.sha,
            });
            console.log(
              "✅ [GEMINI]: main.py Optimized successfully without shrinkage.",
            );
          } else if (shrinkRatio < 80) {
            // AI က Code တွေ ဖြတ်ချလိုက်ရင် ဒီမှာ Block လိုက်ပြီ
            console.log(
              `🚫 [REJECTED]: AI truncated the code! Shrinkage detected. Keeping original main.py.`,
            );
          } else {
            console.log(
              "⚡ [GEMINI]: No optimization required. Code is stable.",
            );
          }
        }
      } else {
        console.log(
          "⚠️ [GEMINI]: Key missing or Power Level 0. Skipping Audit.",
        );
      }
    } catch (auditErr) {
      console.log(
        "🚨 [GEMINI AUDIT FAILED]: Continuing with normal logic...",
        auditErr.message,
      );
    }

    // ၄။ [SELF-EVOLUTION LOGIC] Power 10000 ကျော်လျှင် Core Code ကိုယ်တိုင် Update လုပ်ခြင်း
    if (powerLevel >= 10000) {
      const { data: coreFile } = await octokit.repos.getContent({
        owner: REPO_OWNER,
        repo: CORE_REPO,
        path: "delta_sync.js",
      });

      let content = Buffer.from(coreFile.content, "base64").toString();

      // Evolution Stamp တစ်ခါသာ ရိုက်နှိပ်ရန် စစ်ဆေးခြင်း
      if (!content.includes(`Density: ${powerLevel}`)) {
        const evolvedStamp = `\n// [Natural Order] Last Self-Evolution: ${new Date().toISOString()} | Density: ${powerLevel}`;

        await octokit.repos.createOrUpdateFileContents({
          owner: REPO_OWNER,
          repo: CORE_REPO,
          path: "delta_sync.js",
          message: `🧬 Evolution: Power ${powerLevel}`,
          content: Buffer.from(content + evolvedStamp).toString("base64"),
          sha: coreFile.sha,
        });
        console.log(`🧬 Self-Evolution Successful: Power Level ${powerLevel}`);
      }
    }

    // ၅။ [SWARM BROADCAST] 'neon' client ကိုပါ argument အဖြစ် ထည့်သွင်းပေးခြင်း
    // ဤနေရာတွင် 'neon' ပါမှသာ manageSwarm ထဲ၌ density တိုးပွားခြင်း logic အလုပ်လုပ်မည်ဖြစ်သည်
    await manageSwarm(decision, powerLevel, neon);

    console.log("🏁 MISSION ACCOMPLISHED.");
  } catch (err) {
    console.error("❌ FAILURE:", err.message);
  } finally {
    // Database Connection ကို အမြဲတမ်း ပြန်ပိတ်ပေးရန် (Leak မဖြစ်စေရန်)
    await neon.end();
  }
}

// စနစ်အား စတင်လည်ပတ်စေခြင်း
executeAutonomousTrinity();
