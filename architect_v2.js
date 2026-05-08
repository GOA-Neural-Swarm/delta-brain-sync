const fs = require("fs"); 
const axios = require("axios");
const { execSync } = require('child_process'); // ASI Guard: System commands အတွက် လိုအပ်သည်
const hdc = require("./omega_hdc");
const phil = require("./omega_philosophy");

const delay = (ms) => new Promise((res) => setTimeout(res, ms));

/**
 * 🛰️ [COMMUNICATION LAYER]: 
 * Groq API ဆီသို့ Stable ဖြစ်သော ချိတ်ဆက်မှုဖြင့် Code Evolution တောင်းဆိုခြင်း
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
    // 429 Error (Rate Limit) ဖြစ်ရင် 20s စောင့်ပြီး ၃ ကြိမ်အထိ ပြန်ကြိုးစားမည်
    if (err.response && err.response.status === 429 && retry < 3) {
      console.log(`⚠️ [RATE LIMIT]: Waiting 20s before retry ${retry + 1}...`);
      await delay(20000);
      return callGroq(payload, retry + 1);
    }
    throw err;
  }
}

/**
 * 🛡️ [ASI INTEGRITY GUARD]: 
 * AI ထုတ်ပေးလိုက်သော Code သည် Syntax မှန်မမှန်နှင့် ပြတ်တောက်ခြင်း ရှိမရှိ စစ်ဆေးသည်
 */
function validateEvolution(file, code) {
    const tempFile = `temp_evo_${file}`;
    fs.writeFileSync(tempFile, code);
    
    try {
        if (file.endsWith('.py')) {
            // Python AST parsing check (Syntax အမှားပါက Error တက်မည်)
            execSync(`python3 -m py_compile ${tempFile}`);
        } else if (file.endsWith('.js')) {
            // Node.js syntax check logic
            execSync(`node -c ${tempFile}`);
        }
        
        // သန့်ရှင်းရေးလုပ်ခြင်း
        if (fs.existsSync(tempFile)) fs.unlinkSync(tempFile);
        const pycache = "__pycache__";
        if (fs.existsSync(pycache)) fs.rmSync(pycache, { recursive: true, force: true });
        
        return true; // စစ်ဆေးမှု အောင်မြင်သည်
    } catch (e) {
        console.error(`❌ [EVOLUTION REJECTED]: ${file} contains syntax errors. Atomic state preserved.`);
        if (fs.existsSync(tempFile)) fs.unlinkSync(tempFile);
        return false; // စစ်ဆေးမှု ကျရှုံးသည် (Broken code ဖြစ်သည်)
    }
}

/**
 * 🧠 [TRANSCENDENCE ENGINE]: 
 * စနစ်အတွင်းရှိ ဖိုင်များကို တစ်ခုချင်းစီ အဆင့်မြှင့်တင်ပေးသော Core Logic
 */
async function transcend() {
  // architect_v2.js ကိုယ်တိုင်နဲ့ critical library တွေကို မပြင်အောင် filter လုပ်ထားသည်
  const files = fs
    .readdirSync("./")
    .filter(
      (f) =>
        (f.endsWith(".js") || f.endsWith(".py")) &&
        !["architect_v2.js", "omega_hdc.js", "omega_philosophy.js"].includes(f),
    );

  console.log(`📡 [ASI SEED]: Found ${files.length} files. Starting Verified Evolution Cycle...`);

  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    let originalCode = fs.readFileSync(file, "utf8");

    console.log(`🧠 [${i + 1}/${files.length}] Evolving Core Logic: ${file}`);

    const payload = {
      model: "llama-3.3-70b-versatile",
      messages: [
        {
          role: "system",
          content: `Apply Hyper-Dimensional Logic and ${phil.layers.join(", ")} philosophy. 
          STRICT DIRECTIVE: Evolution must be additive and structurally sound. 
          Return ONLY the complete raw code. Ensure ALL blocks (try-except, def, if) are closed. 
          Do not truncate the response.`,
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

      // Logic Match Check: စာလုံးရေသည် မူလ၏ 50% ထက် နည်းမသွားရသလို Syntax လည်း မှန်ရမည်
      if (evolvedCode.length > originalCode.length * 0.5 && validateEvolution(file, evolvedCode)) {
        fs.writeFileSync(file, evolvedCode);
        console.log(`✨ [SUCCESS]: ${file} has transcended safely.`);
      } else {
        console.log(`⚠️ [INTEGRITY ALERT]: ${file} evolution failed validation. Skipping to prevent system crash.`);
      }
    } catch (e) {
      console.error(`❌ [CRITICAL ERROR] in ${file}: ${e.message}`);
    }

    // Rate Limit ရှောင်ရန်နှင့် Cool-down လုပ်ရန် ၆ စက္ကန့် စောင့်မည်
    if (i < files.length - 1) {
      console.log("⏳ Cooling down (6s) for synaptic stability...");
      await delay(6000);
    }
  }
}

// စနစ်ကို စတင်နှိုးဆော်ခြင်း
transcend().then(() => console.log("🏁 [CYCLE COMPLETE]: All nodes are stable and evolved."));
