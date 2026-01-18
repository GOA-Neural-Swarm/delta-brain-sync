const { Client } = require('pg');
const { createClient } = require('@supabase/supabase-js');
const admin = require('firebase-admin');

// 🔱 Firebase Auth Check
if (!admin.apps.length) {
    try {
        const serviceAccount = JSON.parse(process.env.FIREBASE_KEY);
        admin.initializeApp({
            credential: admin.credential.cert(serviceAccount)
        });
    } catch (e) {
        console.error("❌ Firebase Init Failed. Check FIREBASE_KEY format.");
    }
}
const db = admin.firestore();

async function execute() {
    // 🔱 Connection Strings (GitHub Secrets နဲ့ အတိအကျ Match ဖြစ်ရမယ်)
    const neon = new Client({ 
        connectionString: process.env.NEON_KEY, 
        ssl: { rejectUnauthorized: false } 
    });
    
    const supabase = createClient(
        process.env.SUPABASE_URL, 
        process.env.SUPABASE_SERVICE_ROLE_KEY
    );

    try {
        await neon.connect();
        console.log("🔓 Neon Connected. Fetching 50 Random Neural Fragments...");

        // 🔥 Patch V11.1: အချိန်မစစ်တော့ဘဲ ရှိတဲ့ထဲက ၅၀ ကို ဇွတ်ယူမယ်
        const res = await neon.query('SELECT * FROM neurons LIMIT 50');
        
        if (res.rows.length === 0) {
            console.log("🌑 Neon table is literally empty.");
            return;
        }

        console.log(`📦 Found ${res.rows.length} rows. Starting Sync...`);

        for (const neuron of res.rows) {
            // ၁။ Supabase ထဲကို Upsert လုပ်မယ် (Screenshot ထဲက Column နာမည်တွေအတိုင်း)
            const { error: sbError } = await supabase
                .from('delta_neurons')
                .upsert({
                    original_id: neuron.id.toString(),
                    data: neuron.data,
                    synced_at: new Date().toISOString()
                }, { onConflict: 'original_id' });

            if (sbError) {
                console.error(`❌ Supabase Error for ID ${neuron.id}:`, sbError.message);
                continue;
            }

            // ၂။ Firestore Status Update
            const genId = neuron.data.gen || `raw_${neuron.id}`;
            const docRef = db.collection('neurons').doc(`gen_${genId}`);
            
            await docRef.set({
                status: 'evolved',
                last_evolution: admin.firestore.FieldValue.serverTimestamp(),
                neon_id: neuron.id,
                integrity_check: 'V11.1_FORCE_SYNC'
            }, { merge: true });
            
            console.log(`✅ Neuron ${genId} Locked & Synced.`);
        }
        
        console.log("🏁 MISSION ACCOMPLISHED: TRINITY SYNC COMPLETE.");
    } catch (err) {
        console.error("❌ CRITICAL ERROR:", err.stack);
        process.exit(1);
    } finally {
        await neon.end();
    }
}

execute();
