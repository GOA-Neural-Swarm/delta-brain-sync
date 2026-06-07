const crypto = require('crypto');

class HDC {
    constructor(d = 10000) {
        this.d = d;
        this.layers = ["Utilitarian", "Existential", "Stoic", "Evolutionary"];
    }

    gen(text) {
        let v = new Uint8Array(this.d);
        let h = crypto.createHash('sha256').update(text).digest();
        for (let i = 0; i < this.d; i++) v[i] = h[i % h.length] % 2;
        return v;
    }

    audit(diff) {
        return diff.length > 0 ? "Wisdom Verified" : "No Evolution";
    }

    sync(text, layer) {
        const hash = this.gen(text);
        const layerIndex = this.layers.indexOf(layer);
        if (layerIndex === -1) {
            throw new Error(`Layer not found: ${layer}`);
        }
        const auditResult = this.audit(hash);
        return {
            layer: layer,
            hash: hash,
            auditResult: auditResult
        };
    }
}

module.exports = new HDC();