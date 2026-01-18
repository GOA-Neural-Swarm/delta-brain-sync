const { Client } = require('pg');
const { createClient } = require('@supabase/supabase-js');
const admin = require('firebase-admin');

// 🔱 1. Firebase Engine (Matching Secret: FIREBASE_KEY)
if (!admin.apps.length) {
    try {
        const serviceAccount = JSON.parse(process.env.FIREBASE_KEY);
        admin.initializeApp({
            credential: admin.credential.cert(serviceAccount)
        });
        console.log("🔥 Firebase Engine Ready.");
    } catch (e) {
        console.error("❌ Firebase Secret Error. Check FIREBASE_KEY format.");
        process.exit(1);
    }
}
const db = admin.firestore();

async function executeTrinitySync() {
    // 🔱 2. Database Clients Setup (Match with GitHub Secrets)
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
        console.log("🔓 Neon Core Unlocked. Target Table: neurons");

        // 🔥 Patch V11.1: Fetch 50 raw neurons from Master Table
        const res = await neon.query('SELECT * FROM neurons LIMIT 50');
        console.log(`📡 Processing ${res.rows.length} neural fragments.`);

        if (res.rows.length === 0) {
            console.log("🌑 No neurons found to sync.");
            return;
        }

        for (const neuron of res.rows) {
            // A. Sync to Supabase Master Table ('neurons')
            // Audit အရ 'synced_at' column ကို SQL နဲ့ အရင်တိုးထားဖို့လိုတယ်
            const { error: sbError } = await supabase
                .from('neurons')
                .upsert({
                    id: neuron.id,
                    data: neuron.data,
                    synced_at: new Date().toISOString()
                }, { onConflict: 'id' });

            if (sbError) {
                console.error(`❌ Supabase Sync Error (ID: ${neuron.id}):`, sbError.message);
                continue;
            }

            // B. Firebase Realtime Status Update
            const genId = neuron.data.gen || `raw_${neuron.id}`;
            await db.collection('neurons').doc(`gen_${genId}`).set({
                status: 'evolved',
                neon_id: neuron.id,
                integrity_check: 'V11.1_MASTER_SYNC',
                last_evolution: admin.firestore.FieldValue.serverTimestamp()
            }, { merge: true });

            console.log(`✅ Fragment gen_${genId} Synced & Locked.`);
        }
        
        console.log("🏁 MISSION ACCOMPLISHED: MASTER TRINITY SYNC COMPLETE.");
    } catch (err) {
        console.error("❌ CRITICAL SYSTEM FAILURE:", err.stack);
        process.exit(1);
    } finally {
        await neon.end();
    }
}

// Start the Autonomous Process
executeTrinitySync();
