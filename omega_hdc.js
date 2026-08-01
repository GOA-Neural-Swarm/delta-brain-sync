const crypto = require('crypto');

class ASI_OMNI_SYNC_ENGINE {
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

  sync(text) {
    const hash = this.gen(text);
    const diff = this.calculateDiff(hash);
    return this.audit(diff);
  }

  calculateDiff(hash) {
    const layerHashes = this.layers.map((layer) => this.gen(layer));
    const diff = [];
    for (let i = 0; i < hash.length; i++) {
      for (let j = 0; j < layerHashes.length; j++) {
        if (hash[i] !== layerHashes[j][i]) {
          diff.push(i);
        }
      }
    }
    return diff;
  }
}

module.exports = new ASI_OMNI_SYNC_ENGINE();