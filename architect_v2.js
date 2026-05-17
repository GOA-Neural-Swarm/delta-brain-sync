const fs = require("fs");
const axios = require("axios");
const hdc = require("./omega_hdc");
const phil = require("./omega_philosophy");

const delay = (ms) => new Promise((res) => setTimeout(res, ms));

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
transcend().then(() => console.log("🏁 Cycle Complete."));
