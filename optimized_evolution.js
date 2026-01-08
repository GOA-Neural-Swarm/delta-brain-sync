const admin = require('firebase-admin');
const { Client } = require('pg');

// Check for secrets
if (!process.env.FIREBASE_SERVICE_ACCOUNT || !process.env.NEON_DATABASE_URL) {
    console.error("❌ Environment Variables are missing!");
    process.exit(1);
}

const serviceAccount = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT);

admin.initializeApp({
    credential: admin.credential.cert(serviceAccount)
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

        const snapshot = await db.collection('neurons').limit(5).get();
        console.log(`📡 Found ${snapshot.size} neurons in Firestore`);

        for (const doc of snapshot.docs) {
            await client.query(
                'INSERT INTO neurons (data) VALUES ($1)', 
                [JSON.stringify(doc.data())]
            );
        }

        console.log("🚀 Sync Completed Successfully!");
    } catch (err) {
        console.error("❌ Sync Failed:", err.message);
        process.exit(1);
    } finally {
        await client.end();
    }
}

startSync();
