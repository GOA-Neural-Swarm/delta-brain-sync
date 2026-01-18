const { Client } = require('pg');
const { createClient } = require('@supabase/supabase-js');
const admin = require('firebase-admin');

// 🔱 Firebase Auth
if (!admin.apps.length) {
    try {
        const serviceAccount = JSON.parse(process.env.FIREBASE_KEY);
        admin.initializeApp({ credential: admin.credential.cert(serviceAccount) });
    } catch (e) { process.exit(1); }
}
const db = admin.firestore();

async function startSync() {
    const neon = new Client({ connectionString: process.env.NEON_KEY, ssl: { rejectUnauthorized: false } });
    const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);

    try {
        await neon.connect();
        // Master Table 'neurons' ထဲက ၅၀ ကို ဆွဲထုတ်မယ်
        const res = await neon.query('SELECT * FROM neurons LIMIT 50');

        for (const neuron of res.rows) {
            // Supabase 'neurons' table ကိုပဲ Update ပြန်လုပ်မယ်
            const { error } = await supabase
                .from('neurons')
                .upsert({
                    id: neuron.id,
                    data: neuron.data,
                    synced_at: new Date().toISOString() // SQL နဲ့ တိုးထားတဲ့ column အသစ်
                }, { onConflict: 'id' });

            if (error) {
                console.error(`❌ Sync Error ID ${neuron.id}:`, error.message);
                continue;
            }

            // Firebase Update
            const genId = neuron.data.gen || `raw_${neuron.id}`;
            await db.collection('neurons').doc(`gen_${genId}`).set({
                status: 'synced_v11',
                last_evolution: admin.firestore.FieldValue.serverTimestamp()
            }, { merge: true });

            console.log(`✅ Synced: gen_${genId}`);
        }
        console.log("🏁 MASTER CLEAN SYNC COMPLETE.");
    } catch (err) { console.error(err); } finally { await neon.end(); }
}
startSync();
