const admin = require('firebase-admin');
const { Client } = require('pg');

async function run() {
    // Timeout function: ၁ မိနစ်ကျော်ရင် ဇွတ်ရပ်ခိုင်းမယ်
    const timeout = setTimeout(() => {
        console.error("❌ Timeout: Process took too long!");
        process.exit(1);
    }, 60000);

    try {
        console.log("🚀 Sync Started...");
        admin.initializeApp({
            credential: admin.credential.cert(JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT))
        });
        const db = admin.firestore();

        const client = new Client({
            connectionString: process.env.NEON_DATABASE_URL,
            ssl: { rejectUnauthorized: false }
        });

        await client.connect();
        console.log("✅ Neon Connected!");

        // Table ရှိမရှိ စစ်မယ်၊ မရှိရင် ဆောက်မယ်
        await client.query(`
            CREATE TABLE IF NOT EXISTS neurons (
                id SERIAL PRIMARY KEY,
                data JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        `);

        const snap = await db.collection('neurons').limit(10).get();
        console.log(`📡 Firestore Docs: ${snap.size}`);

        for (const doc of snap.docs) {
            await client.query('INSERT INTO neurons (data) VALUES ($1)', [JSON.stringify(doc.data())]);
        }

        console.log("🏁 SUCCESS: Data Synced!");
        clearTimeout(timeout);
        await client.end();
        process.exit(0);
    } catch (e) {
        console.error("❌ ERROR:", e.message);
        process.exit(1);
    }
}
run();
