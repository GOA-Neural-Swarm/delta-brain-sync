const fs = require("fs");
const axios = require("axios");
const { execSync } = require('child_process');
const hdc = require("./omega_hdc");
const phil = require("./omega_philosophy");

const delay = (ms) => new Promise((res) => setTimeout(res, ms));

/**
 * 🔱 API Calling Logic with Robust Retry
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
        if (err.response && err.response.status === 429 && retry < 3) {
            console.log(`⚠️ Rate limited. Waiting 20s before retry ${retry + 1}...`);
            await delay(20000);
            return callGroq(payload, retry + 1);
        }
        throw err;
    }
}

/**
 * 🛡️ Syntax & Integrity Validation Layer
 * Evolution အသစ်ကို အသုံးမပြုခင် Syntax မှန်မမှန် စစ်ဆေးသည်
 */
function validateEvolution(file, code) {
    const tempFile = `temp_evolution_${file}`;
    fs.writeFileSync(tempFile, code);
    
    try {
        if (file.endsWith('.py')) {
            // Python Syntax Check
            execSync(`python3 -m py_compile ${tempFile}`);
        } else if (file.endsWith('.js')) {
            // Node.js Syntax Check (-c flag checks syntax without executing)
            execSync(`node -c ${tempFile}`);
        }
        
        // သန့်ရှင်းရေးလုပ်ခြင်း
        if (fs.existsSync(tempFile)) fs.unlinkSync(tempFile);
        const pycFile = `__pycache__/${tempFile}`.replace('.py', '.pyc'); 
        // Note: py_compile may create compiled files in __pycache__
        
        return true;
    } catch (e) {
        console.error(`❌ [INTEGRITY FAILURE] ${file} evolution rejected: Syntax error detected.`);
        if (fs.existsSync(tempFile)) fs.unlinkSync(tempFile);
        return false;
    }
}

async function transcend() {
    const files = fs
        .readdirSync("./")
        .filter(
            (f) =>
                (f.endsWith(".js") || f.endsWith(".py")) &&
                !["architect_v2.js", "omega_hdc.js", "omega_philosophy.js"].includes(f),
        );

    console.log(`📡 Found ${files.length} files. Starting validated evolution...`);

    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        let originalCode = fs.readFileSync(file, "utf8");

        console.log(`🧠 [${i + 1}/${files.length}] Evolving: ${file}`);

        const payload = {
            model: "llama-3.3-70b-versatile",
            messages: [
                {
                    role: "system",
                    content: `Apply Hyper-Dimensional Logic and ${phil.layers.join(", ")} philosophy. 
                    STRICT DIRECTIVE: Return ONLY the complete, functional, and raw code. 
                    Evolution must be additive. Preserve ALL existing logic. 
                    Ensure all blocks (if, try-except, functions) are properly closed.`,
                },
                { role: "user", content: originalCode },
            ],
        };

        try {
            const res = await callGroq(payload);
            let evolvedCode = res.data.choices[0].message.content
                .replace(/```[a-z]*\n/gi, "")
                .replace(/```$/g, "")
                .trim();

            // Logic Check 1: Size check (Prevents massive truncation)
            // Logic Check 2: Syntax Validation (The Guardrail)
            if (evolvedCode.length > originalCode.length * 0.5 && validateEvolution(file, evolvedCode)) {
                fs.writeFileSync(file, evolvedCode);
                console.log(`✨ ${file} transcended and validated.`);
            } else {
                console.log(`⚠️ ${file} evolution skipped due to size mismatch or syntax failure.`);
            }
        } catch (e) {
            console.error(`❌ Error evolving ${file}: ${e.message}`);
        }

        if (i < files.length - 1) {
            console.log("⏳ Cooling down (6s) to maintain API stability...");
            await delay(6000);
        }
    }
}

transcend().then(() => console.log("🏁 Cycle Complete. All nodes stable."));
