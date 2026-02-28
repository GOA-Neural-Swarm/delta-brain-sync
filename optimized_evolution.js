const admin = require('firebase-admin');
const { Client } = require('pg');
const fs = require('fs');

/**
 * 📡 [CONTROL LAYER]: Python Core ဆီက လာတဲ့ instruction.json ကို ဖတ်မယ်
 */
function getInstruction() {
    try {
        if (fs.existsSync('instruction.json')) {
            const data = fs.readFileSync('instruction.json', 'utf8');
            return JSON.parse(data);
        }
    } catch (e) {
        console.log("⚠️ [SYSTEM]: No instruction file found, using default.");
    }
    return { command: "NORMAL_GROWTH" }; // ဖိုင်မရှိရင် ပုံမှန်အတိုင်းပဲ သွားမယ်
}

/**
 * 🌀 [EVOLUTION SYNC]: Firebase မှ Neon DB သို့ ဒေတာများ ကူးပြောင်းခြင်း
 */
async function sync() {
    let client; // Neon client ကို အပြင်မှာ ကြေညာထားမယ် (error တက်ရင် ပိတ်နိုင်အောင်)
    
    try {
        console.log("🚀 [STRATEGIC SYNC]: Starting Evolution Cycle...");

        // 1. Python အမိန့်ကို အရင်ဖတ်ပြီး Sync လုပ်မယ့် အရေအတွက် သတ်မှတ်မယ်
        const instr = getInstruction();
        const syncLimit = (instr.command === "HYPER_EXPANSION") ? 50 : 5;
        console.log(`📡 [COMMAND]: ${instr.command} | [LIMIT]: Syncing ${syncLimit} neurons.`);

        // 2. Firebase Initialization
        const serviceAccount = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT);
        if (!admin.apps.length) {
            admin.initializeApp({
                credential: admin.credential.cert(serviceAccount)
            });
        }
        const db = admin.firestore();

        // 3. Neon Database Connection
        client = new Client({
            connectionString: process.env.NEON_DATABASE_URL,
            ssl: { rejectUnauthorized: false }
        });

        await client.connect();
        console.log("✅ [NEON]: Connected successfully.");

        // 4. Database Schema Maintenance (Table ရှိမရှိ စစ်ပြီး လိုအပ်ရင် ဆောက်မယ်)
        await client.query(`
            CREATE TABLE IF NOT EXISTS neurons (
                id SERIAL PRIMARY KEY,
                data JSONB NOT NULL,
                evolved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        `);

        // 5. Data Migration (Firebase -> Neon)
        // syncLimit (၅ သို့မဟုတ် ၅၀) ပေါ်မူတည်ပြီး ဒေတာဆွဲထုတ်မယ်
        const snap = await db.collection('neurons').limit(syncLimit).get();
        
        if (snap.empty) {
            console.log("Empty repository. No neurons to evolve.");
        } else {
            for (const doc of snap.docs) {
                // Firebase က ဒေတာကို Neon ရဲ့ evolved_at column ထဲ ဇွတ်ထည့်မယ့် logic
                await client.query(
                    'INSERT INTO neurons (data, evolved_at) VALUES ($1, NOW())', 
                    [JSON.stringify(doc.data())]
                );
            }
            console.log(`🏁 [SUCCESS]: ${snap.docs.length} neurons manifested on Neon DB!`);
        }

        await client.end();
        console.log("🏁 Mission Accomplished!");

    } catch (err) {
        console.error("❌ [CRITICAL ERROR]:", err.message);
        if (client) await client.end();
        process.exit(1);
    }
}

// 🚀 Execution Start
sync();
