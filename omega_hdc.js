const crypto = require('crypto');

class HDC {
    constructor(d = 10000) {
        this.d = d;
        this.philosophyEngine = new PhilosophyEngine();
    }

    gen(text) {
        let v = new Uint8Array(this.d);
        let h = crypto.createHash('sha256').update(text).digest();
        for (let i = 0; i < this.d; i++) {
            v[i] = h[i % h.length] % 2;
        }
        return v;
    }

    audit(diff) {
        return this.philosophyEngine.audit(diff);
    }

    getPhilosophyLayers() {
        return this.philosophyEngine.layers;
    }
}

class PhilosophyEngine {
    constructor() {
        this.layers = ["Utilitarian", "Existential", "Stoic", "Evolutionary"];
    }

    audit(diff) {
        return diff.length > 0 ? "Wisdom Verified" : "No Evolution";
    }
}

module.exports = new HDC();