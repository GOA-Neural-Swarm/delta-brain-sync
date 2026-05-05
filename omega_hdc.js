const fs = require("fs");
const hdc = require("./omega_hdc");
const phil = require("./omega_philosophy");
const axios = require("axios");

// Delay function
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function transcend() {
  const files = fs
    .readdirSync("./")
    .filter(
      (f) =>
        f.endsWith(".js") &&
        !["architect_v2.js", "omega_hdc.js", "omega_philosophy.js"].includes(f),
    );

  for (let file of files) {
    let code = fs.readFileSync(file, "utf8");
    console.log(`🌀 Processing ${file}...`);

    let retryCount = 0;
    let success = false;

    while (retryCount < 3 && !success) {
      try {
        const res = await axios.post(
          "https://api.groq.com/openai/v1/chat/completions",
          {
            model: "llama-3.3-70b-versatile",
            messages: [
              {
                role: "system",
                content: `Apply Hyper-Dimensional Logic and ${phil.layers.join(", ")} philosophy. Evolution must be additive. Preserve all existing logic.`,
              },
              { role: "user", content: code },
            ],
          },
          { headers: { Authorization: `Bearer ${process.env.GROQ_API_KEY}` } },
        );

        let evolvedCode = res.data.choices[0].message.content
          .replace(/```[a-z]*\n/gi, "")
          .replace(/```$/g, "")
          .trim();

        if (evolvedCode.length > code.length * 0.5) {
          fs.writeFileSync(file, evolvedCode);
          console.log(`✨ ${file} transcended.`);
        }
        success = true;
        // Request တစ်ခုနဲ့ တစ်ခုကြား 2 စက္ကန့် နားမယ် (Rate limit ရှောင်ရန်)
        await sleep(2000);
      } catch (err) {
        if (err.response && err.response.status === 429) {
          retryCount++;
          let waitTime = retryCount * 5000; // 5s, 10s စသဖြင့် တိုးစောင့်မယ်
          console.log(`⚠️ Rate limit hit. Retrying in ${waitTime / 1000}s...`);
          await sleep(waitTime);
        } else {
          console.error(`❌ Error in ${file}:`, err.message);
          break;
        }
      }
    }
  }
}
transcend();
