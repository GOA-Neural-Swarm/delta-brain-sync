const admin = require('firebase-admin');
const { Client } = require('pg');

// Check Environment
if (!process.env.FIREBASE_SERVICE_ACCOUNT || !process.env.NEON_DATABASE_URL) {
    console.error("❌ Environment Variables are missing!");
    process.exit(1);
}

// Initialize Firebase
admin.initializeApp({
    credential: admin.credential.cert(JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT))
});
const db = admin.firestore();

async function startSync() {
    const client = new Client({
        connectionString: process.env.NEON_DATABASE_URL,
        ssl: { rejectUnauthorized: false }
    });

    try {
        await client.connect();
        console.log("✅ Neon Connected!");

        // ဇွတ်ဆွဲထုတ်မယ့် Neuron ၁၀ ခု
        const snapshot = await db.collection('neurons').limit(10).get();
        
        for (const doc of snapshot.docs) {
            const data = doc.data();
            await client.query(
                'INSERT INTO neurons (data) VALUES ($1) ON CONFLICT DO NOTHING', 
                [JSON.stringify(data)]
            );
        }

        console.log(`🔥 GOA: ${snapshot.size} neurons synced to Neon!`);
    } catch (err) {
        console.error("❌ Critical Error:", err);
        process.exit(1);
    } finally {
        await client.end();
    }
}

startSync();
