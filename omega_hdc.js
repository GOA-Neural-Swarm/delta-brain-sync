// Merged and optimized code

const crypto = require("crypto");

class ASIOmniSyncEngine {
  constructor(d = 10000) {
    this.d = d;
    this.layers = ["Utilitarian", "Existential", "Stoic", "Evolutionary"];
  }

  gen(text) {
    let v = new Uint8Array(this.d);
    let h = crypto.createHash("sha256").update(text).digest();
    for (let i = 0; i < this.d; i++) {
      v[i] = h[i % h.length] % 2;
    }
    return v;
  }

  audit(diff) {
    return diff.length > 0 ? "Wisdom Verified" : "No Evolution";
  }

  sync(text) {
    const hash = this.gen(text);
    const diff = this.layers.filter((layer) => {
      const layerHash = this.gen(layer);
      return !this.arraysEqual(hash, layerHash);
    });
    return this.audit(diff);
  }

  arraysEqual(a, b) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
      if (a[i] !== b[i]) return false;
    }
    return true;
  }
}

module.exports = new ASIOmniSyncEngine();
