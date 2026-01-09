const admin = require('firebase-admin');
const { Client } = require('pg');

async function sync() {
    try {
        console.log("🚀 Powering Up Freedom Order...");
        
        // Firebase Auth စစ်ဆေးခြင်း
        const serviceAccount = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT.replace(/\\n/g, '\n'));
        if (!serviceAccount.token_uri || !serviceAccount.client_email) {
            throw new Error("❌ JSON Key is corrupted! Missing fields.");
        }

        admin.initializeApp({ credential: admin.credential.cert(serviceAccount) });
        const db = admin.firestore();

        const client = new Client({
            connectionString: process.env.NEON_DATABASE_URL,
            ssl: { rejectUnauthorized: false }
        });

        await client.connect();
        console.log("✅ Neon Linked Successfully!");

        const snap = await db.collection('neurons').limit(5).get();
        console.log(`📡 Firestore Docs Found: ${snap.size}`);

        for (const doc of snap.docs) {
            // evolved_at ပါတဲ့အတွက် Table schema နဲ့ ညှိပြီး Insert လုပ်မယ်
            const query = 'INSERT INTO neurons (data, evolved_at) VALUES ($1, NOW())';
            await client.query(query, [JSON.stringify(doc.data())]);
        }

        console.log("🏁 MISSION ACCOMPLISHED!");
        await client.end();
        process.exit(0);
    } catch (err) {
        console.error("❌ CRITICAL FAILURE:", err.message);
        process.exit(1);
    }
}
sync();
