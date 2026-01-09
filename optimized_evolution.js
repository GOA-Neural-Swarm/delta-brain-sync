const admin = require('firebase-admin');
const { Client } = require('pg');

async function sync() {
    try {
        console.log("🚀 Strategic Sync Starting...");
        const serviceAccount = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT);
        
        admin.initializeApp({ credential: admin.credential.cert(serviceAccount) });
        const db = admin.firestore();

        const client = new Client({
            connectionString: process.env.NEON_DATABASE_URL,
            ssl: { rejectUnauthorized: false }
        });

        await client.connect();
        console.log("✅ Neon Connected!");

        // Table ရှိမရှိစစ်ပြီး မရှိရင် ဇွတ်ဆောက်မယ်
        await client.query(`
            CREATE TABLE IF NOT EXISTS neurons (
                id SERIAL PRIMARY KEY,
                data JSONB NOT NULL,
                evolved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        `);

        const snap = await db.collection('neurons').limit(5).get();
        for (const doc of snap.docs) {
            // Colab မှာတွေ့တဲ့ evolved_at column ထဲကို ဇွတ်ထည့်မယ်
            await client.query('INSERT INTO neurons (data, evolved_at) VALUES ($1, NOW())', [JSON.stringify(doc.data())]);
        }

        console.log("🏁 SUCCESS: Mission Accomplished!");
        await client.end();
    } catch (err) {
        console.error("❌ ERROR:", err.message);
        process.exit(1);
    }
}
sync();
