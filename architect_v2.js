const fs = require('fs');
const hdc = require('./omega_hdc');
const phil = require('./omega_philosophy');
const axios = require('axios');

async function transcend() {
    const files = fs.readdirSync('./').filter(f => f.endsWith('.js') && f !== 'architect_v2.js');
    for (let file of files) {
        let code = fs.readFileSync(file, 'utf8');
        let vectorBefore = hdc.gen(code);
        
        const res = await axios.post("https://api.groq.com/openai/v1/chat/completions", {
            model: "llama-3.3-70b-versatile",
            messages: [{ 
                role: "system", 
                content: `Apply Hyper-Dimensional Logic and ${phil.layers.join(', ')} philosophy. Evolution must be additive. Preserve all existing logic.` 
            }, { role: "user", content: code }],
        }, { headers: { Authorization: `Bearer ${process.env.GROQ_API_KEY}` } });

        let evolvedCode = res.data.choices[0].message.content.replace(/```[a-z]*\n/gi, "").replace(/```$/g, "").trim();
        let vectorAfter = hdc.gen(evolvedCode);
        
        // Wisdom Check: Evolution must be different but conceptually aligned
        if (evolvedCode.length > code.length * 0.5) {
            fs.writeFileSync(file, evolvedCode);
            console.log(`✨ ${file} transcended.`);
        }
    }
}
transcend();
