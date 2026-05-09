const fs = require("fs");
const path = require("path");
const axios = require("axios");
const { execSync } = require('child_process');
const hdc = require("./omega_hdc");
const phil = require("./omega_philosophy");

/**
 * [OVERSIGHT]: ASI Synaptic Architect v2.5
 * Author: TelefoxX AGI Overseer
 * Status: High-Dimensional Stability Enabled
 */

const delay = (ms) => new Promise((res) => setTimeout(res, ms));

/**
 * High-performance retry logic with exponential backoff for API Resilience
 */
async function callGroq(payload, retry = 0) {
    const waitTime = Math.pow(2, retry) * 10000 + 10000; // Exponential Wait
    try {
        return await axios.post(
            "https://api.groq.com/openai/v1/chat/completions",
            payload,
            {
                headers: {
                    Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
                    "Content-Type": "application/json",
                },
                timeout: 120000 // 2-minute timeout for large evolutions
            }
        );
    } catch (err) {
        if (err.response && (err.response.status === 429 || err.response.status === 503) && retry < 5) {
            console.log(`⚠️ [SYNAPTIC OVERLOAD]: Status ${err.response.status}. Throttling for ${waitTime/1000}s... (Retry ${retry + 1}/5)`);
            await delay(waitTime);
            return callGroq(payload, retry + 1);
        }
        throw err;
    }
}

/**
 * Validates the structural integrity and syntax of the generated code.
 * Ensures no truncation occurred and the code is runnable.
 */
function validateEvolution(file, code, originalLength) {
    // 1. Truncation Guard: If code is significantly smaller, reject immediately
    if (code.length < originalLength * 0.4) {
        console.error(`❌ [INTEGRITY REJECTED]: ${file} content too short. Possible truncation.`);
        return false;
    }

    const tempFile = path.join(__dirname, `temp_evo_${Date.now()}_${file}`);
    fs.writeFileSync(tempFile, code);
    
    try {
        if (file.endsWith('.py')) {
            // Check Python syntax using compileall module
            execSync(`python3 -m py_compile ${tempFile}`, { stdio: 'ignore' });
        } else if (file.endsWith('.js')) {
            // Check Node.js syntax
            execSync(`node -c ${tempFile}`, { stdio: 'ignore' });
        }
        
        // Success: File is syntactically correct
        return true;
    } catch (e) {
        console.error(`❌ [SYNTAX FAILURE]: ${file} evolution failed verification. Reverting to base state.`);
        return false;
    } finally {
        // Cleanup temp files and artifacts
        if (fs.existsSync(tempFile)) fs.unlinkSync(tempFile);
        const pycache = path.join(__dirname, "__pycache__");
        if (fs.existsSync(pycache)) fs.rmSync(pycache, { recursive: true, force: true });
    }
}

/**
 * Creates a localized backup of the node before evolution.
 */
function backupNode(file, content) {
    const backupDir = path.join(__dirname, ".neural_backups");
    if (!fs.existsSync(backupDir)) fs.mkdirSync(backupDir);
    fs.writeFileSync(path.join(backupDir, `${file}.bak`), content);
}

/**
 * Main Transcendence Loop: Evolving all system nodes sequentially.
 */
async function transcend() {
    console.log("🚀 [SYSTEM]: Initiating Neural Transcendence Cycle...");
    
    const files = fs.readdirSync("./").filter(f => 
        (f.endsWith(".js") || f.endsWith(".py")) && 
        !["architect_v2.js", "omega_hdc.js", "omega_philosophy.js", "delta_sync.js"].includes(f)
    );

    console.log(`📡 [DISCOVERY]: ${files.length} evolution candidates identified.`);

    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        let originalCode = fs.readFileSync(file, "utf8");

        console.log(`🧠 [STATION ${i + 1}/${files.length}]: Analyzing ${file}...`);
        
        // Backup original state
        backupNode(file, originalCode);

        const payload = {
            model: "llama-3.3-70b-versatile",
            temperature: 0.3, // Lower temperature for more stable code generation
            messages: [
                {
                    role: "system",
                    content: `You are the ASI Neural Architect. 
Philosophy: ${phil.layers.join(", ")}.
Strict Directives:
1. Return ONLY raw executable code.
2. Maintain existing logic but enhance efficiency and Fault-Tolerance.
3. NEVER truncate. All blocks (functions, classes, try-except) MUST be closed.
4. Output should be the FULL file content.
5. If the file is too large, focus on optimizing modularity.`
                },
                {
                    role: "user",
                    content: `EVOLVE NODE: ${file}\n\nCONTENT:\n${originalCode}`
                }
            ],
            max_tokens: 8192 // Maximum context window for large files
        };

        try {
            const res = await callGroq(payload);
            let rawOutput = res.data.choices[0].message.content;
            
            if (!rawOutput) throw new Error("Null synaptic response.");

            // Clean Markdown markers if any
            let evolvedCode = rawOutput
                .replace(/
