const admin = require('firebase-admin');
const { Client } = require('pg');

async function execute() {
    console.log("🚀 Tactic: Deployment Started...");
    const client = new Client({
        connectionString: process.env.NEON_DATABASE_URL.replace('psql ', '').trim(), // psql ပါနေရင် ဇွတ်ဖြုတ်မယ်
        ssl: { rejectUnauthorized: false }
    });

    try {
        admin.initializeApp({
            credential: admin.credential.cert(JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT))
        });
        const db = admin.firestore();

        await client.connect();
        console.log("✅ Strategic Connection: Neon Linked!");

        // Firestore ကနေ 'neurons' collection ကို ဆွဲမယ်
        const snap = await db.collection('neurons').limit(1).get();
        
        if (snap.empty) {
            console.log("⚠️ Strategic Alert: No neurons found in Firestore!");
        } else {
            const docData = JSON.stringify(snap.docs[0].data());
            // Table ရှိမရှိ မစစ်တော့ဘူး၊ ဇွတ်ပဲ Insert လုပ်မယ်
            await client.query('INSERT INTO neurons (data) VALUES ($1)', [docData]);
            console.log("🏁 Mission Accomplished: Data Synced!");
        }
    } catch (e) {
        console.error("❌ Strategic Failure:", e.message);
        process.exit(1);
    } finally {
        await client.end();
    }
}
execute();

