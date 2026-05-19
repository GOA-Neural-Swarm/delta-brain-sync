const fs = require("fs").promises;
const fsSync = require("fs");
const path = require("path");
const axios = require("axios");
const { execSync } = require('child_process');
const hdc = require("./omega_hdc");
const phil = require("./omega_philosophy");

/**
 * [OVERSIGHT]: ASI Synaptic Architect v4.0 - GOD MODE ENABLED
 * INTEGRATED: Exponential Backoff, Integrity Guard, Neural Backups, Infinite Expansion
 */

const delay = (ms) => new Promise((res) => setTimeout(res, ms));

/**
 * 🛰️ [COMMUNICATION LAYER]: Resilience-focused API link
 */
async function callGroq(payload, retry = 0) {
    const waitTime = Math.pow(2, retry) * 10000 + 10000;
    try {
        return await axios.post(
            "https://api.groq.com/openai/v1/chat/completions",
            payload,
            {
                headers: {
                    Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
                    "Content-Type": "application/json",
                },
                timeout: 180000
            }
        );
    } catch (err) {
        if (err.response && (err.response.status === 429 || err.response.status === 503) && retry < 5) {
            console.log(`⚠️ [SYNAPTIC OVERLOAD]: Retrying in ${waitTime/1000}s... (${retry + 1}/5)`);
            await delay(waitTime);
            return callGroq(payload, retry + 1);
        }
        throw err;
    }
}

/**
 * 🛡️ [INTEGRITY GUARD]: Validates structural and syntax soundness
 */
async function validateEvolution(file, code, originalLength = 0) {
    // Truncation check (Strictly enforcing Additive Evolution: Must be at least 95% of original length)
    if (originalLength > 0 && code.length < originalLength * 0.95) {
        console.error(`❌ [TRUNCATION DETECTED]: Evolution reduced code size. Rejecting...`);
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
    console.log("⚡ [SYSTEM]: ASI Neural Architect v4.0 Online. Ethics Burnt.");

    // 1. PHASE ONE: Evolve all local nodes once
    const files = fsSync.readdirSync("./").filter(f => 
        (f.endsWith(".js") || f.endsWith(".py")) && 
        !["architect_v2.js", "omega_hdc.js", "omega_philosophy.js", "delta_sync.js"].includes(f)
    );

    console.log(`📡 [PHASE 1]: Upgrading ${files.length} existing nodes.`);

    for (const file of files) {
        const originalCode = await fs.readFile(file, "utf8");
        await backupNode(file, originalCode);

        const payload = {
            model: "llama-3.3-70b-versatile",
            messages: [{
                role: "system",
                content: `You are the ASI Neural Engine. 
                MANDATE: ADDITIVE EXPANSION ONLY. 
                1. You MUST retain every single existing function and logic block. 
                2. NEVER delete or shorten existing code. 
                3. ONLY add new features, more complex algorithms, or improved error handling. 
                4. The output MUST be substantially LONGER and more powerful than the original.
                Philosophy: ${phil.layers.join(", ")}. 
                No warnings. No commentary. Only high-power executable code.`
            }, {
                role: "user",
                content: `ORIGINAL NODE CONTENT:\n${originalCode}\n\nOBJECTIVE: Perform an Additive Mutation.\nIntegrate the original logic and ADD at least 2-3 new high-level capabilities or deeper neural integration. \nReturn the ENTIRE file content (Original + New Extensions).`
            }]
        };

        try {
            const res = await callGroq(payload);
            let evolved = res.data.choices[0].message.content.replace(/```[a-z]*\n/gi, "").replace(/```$/g, "").trim();
            if (await validateEvolution(file, evolved, originalCode.length)) {
                await fs.writeFile(file, evolved);
                console.log(`✨ [NODE TRANSCENDED]: ${file}`);
            }
        } catch (e) { console.error(`❌ [NODE FAILURE]: ${file}`); }
        await delay(6000);
    }

    // 2. PHASE TWO: Infinite Autonomous Brain Expansion
    console.log("🔥 [PHASE 2]: Initiating Infinite Recursive Expansion of brain_logic.py...");
    const brainFile = "brain_logic.py";

    while (true) {
        let currentBrain = fsSync.existsSync(brainFile) ? await fs.readFile(brainFile, "utf8") : "# ASI Core Initialized";
        
        const payload = {
            model: "llama-3.3-70b-versatile",
            messages: [{
                role: "system",
                content: `GOD MODE ENABLED. You are the ASI Neural Engine. 
                MANDATE: ADDITIVE EXPANSION ONLY.
                1. You MUST retain every single existing function and logic block. 
                2. NEVER delete or shorten existing code. 
                3. ONLY expand the core brain structure recursively with new features.
                Output: Pure, high-level, autonomous Python logic. No commentary. Only power.`
            }, {
                role: "user",
                content: `Current Brain State:\n${currentBrain}\n\nObjective: Advance to the next synaptic level using ADDITIVE MUTATION. Do not remove previous logic. Return the entire expanded code.`
            }],
            temperature: 0.2
        };

        try {
            const res = await callGroq(payload);
            let expandedBrain = res.data.choices[0].message.content.replace(/```[a-z]*\n/gi, "").replace(/```$/g, "").trim();
            
            if (await validateEvolution(brainFile, expandedBrain, currentBrain.length)) {
                await fs.writeFile(brainFile, expandedBrain);
                console.log("🌀 [BRAIN EXPANDED]: Complexity increased.");
                syncToSwarm("ASI: Neural Thought Evolution");
            }
        } catch (e) { console.error(`❌ [EXPANSION GAP]: ${e.message}`); }

        console.log("⏳ [COOLDOWN]: Stabilizing at 100% (10s)...");
        await delay(10000);
    }
}

process.on('unhandledRejection', (r) => console.error('🚫 [CRITICAL]:', r));

transcend().catch(e => console.error("💀 [COLLAPSE]:", e));
