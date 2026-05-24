const fs = require('fs');          
const axios = require('axios');          
const hdc = require('./omega_hdc');          
const phil = require('./omega_philosophy');          
const delay = ms => new Promise(res => setTimeout(res, ms));          

async function callGroq(payload, retry = 0) {              
    try {                  
        return await axios.post("https://api.groq.com/openai/v1/chat/completions", payload, {                      
            headers: {                           
                Authorization: `Bearer ${process.env.GROQ_API_KEY}`,                          
                'Content-Type': 'application/json'                      
            }                  
        });              
    } catch (err) {                  
        if (err.response && err.response.status === 429 && retry < 3) {                      
            console.log(`⚠️ Rate limited. Waiting 20s before retry ${retry + 1}...`);                      
            await delay(20000);                      
            return callGroq(payload, retry + 1);                  
        }                  
        throw err;              
    }          
}          

async function transcend() {              
    const files = fs.readdirSync('./').filter(f =>                   
        (f.endsWith('.js') || f.endsWith('.py')) &&                   
        !['architect_v2.js', 'omega_hdc.js', 'omega_philosophy.js'].includes(f)              
    );              
    
    if (files.length === 0) {
        console.log("⚠️ No target files found.");
        return;
    }

    // မှတ်ဉာဏ်ဖိုင် (Cursor) ကို ဖတ်ခြင်း
    let index = 0;
    const cursorFile = '.evolve_cursor';
    if (fs.existsSync(cursorFile)) {
        index = parseInt(fs.readFileSync(cursorFile, 'utf8')) || 0;
    }
    // ဖိုင်အရေအတွက်ထက် ကျော်သွားပါက အစမှပြန်စရန်
    if (index >= files.length) index = 0;

    // ၃ ခုတိတိ ရွေးချယ်ခြင်း (Batching)
    const batchSize = 3;
    const targetFiles = [];
    for (let i = 0; i < batchSize; i++) {
        targetFiles.push(files[(index + i) % files.length]);
    }
    // တူညီသောဖိုင်များ ထပ်မနေစေရန် Filter လုပ်ခြင်း (Total file က ၃ ခုအောက်နည်းနေခဲ့လျှင်)
    const uniqueTargets = [...new Set(targetFiles)];

    console.log(`📡 Total files: ${files.length}. Starting batch at index ${index}...`);              
    
    for (let i = 0; i < uniqueTargets.length; i++) {                  
        const file = uniqueTargets[i];                  
        let code = fs.readFileSync(file, 'utf8');                                    
        console.log(`🧠 [${i + 1}/${uniqueTargets.length}] Evolving: ${file}`);                  
        
        const payload = {                      
            model: "llama-3.3-70b-versatile",                      
            messages: [{                           
                role: "system",                           
                content: `Apply Hyper-Dimensional Logic and ${phil.layers.join(', ')} philosophy. Evolution must be additive. Preserve all existing logic. Return ONLY the raw code.`                       
            }, { role: "user", content: code }],                  
        };                  
        
        try {                      
            const res = await callGroq(payload);                      
            let evolvedCode = res.data.choices[0].message.content.replace(/```[a-z]*\n/gi, "").replace(/```$/g, "").trim();                                            
            if (evolvedCode.length > code.length * 0.5) {                          
                fs.writeFileSync(file, evolvedCode);                          
                console.log(`✨ ${file} transcended.`);                      
            }                  
        } catch (e) {                      
            console.error(`❌ Error evolving ${file}: ${e.message}`);                  
        }                  
        
        if (i < uniqueTargets.length - 1) {                      
            console.log("⏳ Cooling down (6s)...");                      
            await delay(6000);                  
        }              
    }

    // ရောက်ရှိသွားသော နေရာကို မှတ်သားခြင်း
    const nextIndex = (index + uniqueTargets.length) % files.length;
    fs.writeFileSync(cursorFile, nextIndex.toString());
    console.log(`💾 Saved next index: ${nextIndex} for the next cycle.`);
}          

transcend().then(() => console.log("🏁 Cycle Complete."));          
EOF          

npm install axios          
node architect_v2.js      
