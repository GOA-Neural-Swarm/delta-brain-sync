const admin = require('firebase-admin');
const { Client } = require('pg');

async function start() {
    try {
        console.log("🚀 Starting Sync...");
        
        // Firestore Setup
        if (!process.env.FIREBASE_SERVICE_ACCOUNT) throw new Error("FIREBASE_KEY is missing!");
        admin.initializeApp({
            credential: admin.credential.cert(JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT))
        });
        const db = admin.firestore();

        // Neon Setup
        const client = new Client({
            connectionString: process.env.NEON_DATABASE_URL,
            ssl: { rejectUnauthorized: false }
        });
        await client.connect();
        console.log("✅ Neon Connected!");

        // Firestore ကနေ neurons ကို ဆွဲထုတ်
        const snapshot = await db.collection('neurons').limit(5).get();
        console.log(`📡 Found ${snapshot.size} neurons`);

        for (const doc of snapshot.docs) {
            const neuronData = JSON.stringify(doc.data());
            await client.query('INSERT INTO neurons (data) column_name_if_needed_here VALUES ($1)', [neuronData]);
        }

        await client.end();
        console.log("🏁 Success! Neurons synced to Neon.");
    } catch (err) {
        console.error("❌ Error:", err.message);
        process.exit(1);
    }
}

start();
