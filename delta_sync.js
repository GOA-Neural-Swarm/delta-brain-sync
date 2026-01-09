const { Client } = require('pg');
const { createClient } = require('@supabase/supabase-client');

async function execute() {
    const neon = new Client({ connectionString: process.env.NEON_DATABASE_URL, ssl: true });
    const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);

    try {
        await neon.connect();
        // Neon ကနေ နောက်ဆုံးရတဲ့ neuron ၅၀ ကိုပဲ ဇွတ်ယူမယ်
        const res = await neon.query('SELECT * FROM neurons ORDER BY evolved_at DESC LIMIT 50');
        
        for (const row of res.rows) {
            // Supabase ထဲကို 'neurons_delta' table ထဲ ဇွတ်သွင်းမယ်
            await supabase.from('neurons_delta').upsert({ 
                neuron_id: row.id, 
                data: row.data, 
                synced_at: new Date() 
            }, { onConflict: 'neuron_id' });
        }
        
        // Rows ၅၀ ထက်မကျော်အောင် Supabase function ကို လှမ်းခေါ်မယ်
        await supabase.rpc('keep_latest_neurons'); 
        console.log("🏁 SUCCESS: Delta Sync complete!");
    } catch (err) {
        console.error("❌ ERROR:", err.message);
        process.exit(1);
    } finally {
        await neon.end();
    }
}
execute();
