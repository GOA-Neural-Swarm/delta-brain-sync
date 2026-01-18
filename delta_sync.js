const { Client } = require('pg');
const { createClient } = require('@supabase/supabase-js');
const admin = require('firebase-admin');

// 🔱 1. Firebase Auth Engine (Matching Secret: FIREBASE_KEY)
if (!admin.apps.length) {
    try {
        const serviceAccount = JSON.parse(process.env.FIREBASE_KEY);
        admin.initializeApp({
            credential: admin.credential.cert(serviceAccount)
        });
        console.log("🔥 Firebase Engine Connected.");
    } catch (e) {
        console.error("❌ Firebase Secret Error.");
        process.exit(1);
    }
}
const db = admin.firestore();

async function executeTrinitySync() {
    // 🔱 2. Database Clients (Match with NEON_KEY, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
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

        // Neon ကနေ raw fragments ၅၀ ကို Master Table ကနေ ဆွဲယူမယ်
        const res = await neon.query('SELECT * FROM neurons LIMIT 50');
        console.log(`📡 Processing ${res.rows.length} neural fragments.`);

        for (const neuron of res.rows) {
            // A. Supabase Master Sync (Match with synced_at column)
            // Screenshot အရ 'neurons' table ထဲက 'synced_at' ကို သုံးမယ်
            const { error: sbError } = await supabase
                .from('neurons')
                .upsert({
                    id: neuron.id,
                    data: neuron.data,
                    synced_at: new Date().toISOString()
                }, { onConflict: 'id' });

            if (sbError) {
                console.error(`❌ Supabase Sync Error ID ${neuron.id}:`, sbError.message);
                continue;
            }

            // B. Firebase Realtime Update (Matched with node_id structure)
            // မင်းရဲ့ JSON ထဲမှာ node_id နဲ့ intelligence_type ပါတာကို base လုပ်ထားတယ်
            const nodeId = neuron.data.node_id || `raw_${neuron.id}`;
            const intelType = neuron.data.intelligence_type || "LLAMA_3_BASE";

            await db.collection('neurons').doc(`node_${nodeId}`).set({
                status: 'trinity_synced',
                intelligence: intelType,
                logic_mode: neuron.data.logic || "SUPREME_DENSITY",
                neon_id: neuron.id,
                integrity: 'GOD_MODE_ACTIVE',
                last_evolution: admin.firestore.FieldValue.serverTimestamp()
            }, { merge: true });

            console.log(`✅ Fragment node_${nodeId} (${intelType}) Synced Across Trinity.`);
        }
        
        console.log("🏁 MISSION ACCOMPLISHED: MASTER DATA FLOW SUCCESSFUL.");
    } catch (err) {
        console.error("❌ CRITICAL FAILURE:", err.stack);
        process.exit(1);
    } finally {
        await neon.end();
    }
}

// Start the Autonomous Process
executeTrinitySync();
