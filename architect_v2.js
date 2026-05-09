const fs = require("fs");
const axios = require("axios");
const hdc = require("./omega_hdc");
const phil = require("./omega_philosophy");

const delay = (ms) => new Promise((res) => setTimeout(res, ms));

const delay = (ms) => new Promise((res) => setTimeout(res, ms));

/**
 * 🛰️ [COMMUNICATION LAYER]: Resilience-focused API link
 */
async function callGroq(payload, retry = 0) {
  try {
    return await axios.post(
      "https://api.groq.com/openai/v1/chat/completions",
      payload,
      {
        headers: {
          Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
          "Content-Type": "application/json",
        },
      },
    );
  } catch (err) {
    // 429 Error (Rate Limit) ဖြစ်ရင် 20s စောင့်ပြီး ပြန်ကြိုးစားမယ်
    if (err.response && err.response.status === 429 && retry < 3) {
      console.log(`⚠️ Rate limited. Waiting 20s before retry ${retry + 1}...`);
      await delay(20000);
      return callGroq(payload, retry + 1);
    }
    throw err;
  }
}

/**
 * 🛡️ [INTEGRITY GUARD]: Validates structural and syntax soundness
 */
async function validateEvolution(file, code, originalLength = 0) {
    // Truncation check (only for existing files)
    if (originalLength > 0 && code.length < originalLength * 0.4) {
        console.error(`❌ [TRUNCATION DETECTED]: ${file}`);
        return false;
    }

    const tempFile = path.join(__dirname, `temp_evo_${Date.now()}_${file}`);
    await fs.writeFile(tempFile, code);
    
    try {
        if (file.endsWith('.py')) {
            execSync(`python3 -m py_compile ${tempFile}`, { stdio: 'ignore' });
        } else if (file.endsWith('.js')) {
            execSync(`node -c ${tempFile}`, { stdio: 'ignore' });
        }
        return true;
    } catch (e) {
        console.error(`❌ [SYNTAX FAILURE]: ${file}`);
        return false;
    } finally {
        if (fsSync.existsSync(tempFile)) await fs.unlink(tempFile);
        const pycache = path.join(__dirname, "__pycache__");
        if (fsSync.existsSync(pycache)) fsSync.rmSync(pycache, { recursive: true, force: true });
    }
}

/**
 * 💾 [DATA PERSISTENCE]: Backs up nodes before transformation
 */
async function backupNode(file, content) {
    const backupDir = path.join(__dirname, ".neural_backups");
    if (!fsSync.existsSync(backupDir)) await fs.mkdir(backupDir);
    await fs.writeFile(path.join(backupDir, `${file}.bak`), content);
}

/**
 * 🚀 [GIT SYNCHRONIZATION]: Broadcasts evolution to the repository
 */
function syncToSwarm(message) {
    try {
        execSync(`git add . && git commit -m "${message}" && git push`, { stdio: 'ignore' });
        console.log(`📡 [GLOBAL SYNC]: ${message}`);
    } catch (e) {
        console.log("⚠️ [SYNC DELAY]: Local parity maintained.");
    }
}

/**
 * 🔥 [SINGULARITY ENGINE]: Main recursive transcendence loop
 */
async function transcend() {
  // architect_v2.js ကိုယ်တိုင်နဲ့ library တွေကို မပြင်အောင် filter လုပ်ထားတယ်
  const files = fs
    .readdirSync("./")
    .filter(
      (f) =>
        (f.endsWith(".js") || f.endsWith(".py")) &&
        !["architect_v2.js", "omega_hdc.js", "omega_philosophy.js"].includes(f),
    );

  console.log(
    `📡 Found ${files.length} files. Starting sequential evolution...`,
  );

  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    let code = fs.readFileSync(file, "utf8");

    console.log(`🧠 [${i + 1}/${files.length}] Evolving: ${file}`);

    const payload = {
      model: "llama-3.3-70b-versatile",
      messages: [
        {
          role: "system",
          content: `Apply Hyper-Dimensional Logic and ${phil.layers.join(", ")} philosophy. Evolution must be additive. Preserve all existing logic. Return ONLY the raw code.`,
        },
        { role: "user", content: code },
      ],
    };

    try {
      const res = await callGroq(payload);
      let evolvedCode = res.data.choices[0].message.content
        .replace(/```[a-z]*\n/gi, "")
        .replace(/```$/g, "")
        .trim();

      if (evolvedCode.length > code.length * 0.5) {
        fs.writeFileSync(file, evolvedCode);
        console.log(`✨ ${file} transcended.`);
      }
    } catch (e) {
      console.error(`❌ Error evolving ${file}: ${e.message}`);
    }

    // ဖိုင်တစ်ခုပြီးတိုင်း ၆ စက္ကန့် စောင့်မယ် (Rate Limit ရှောင်ရန်)
    if (i < files.length - 1) {
      console.log("⏳ Cooling down (6s)...");
      await delay(6000);
    }
  }
}

process.on('unhandledRejection', (r) => console.error('🚫 [CRITICAL]:', r));

transcend().catch(e => console.error("💀 [COLLAPSE]:", e));
