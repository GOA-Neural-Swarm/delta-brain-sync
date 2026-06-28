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

  mergeSync(node) {
    if (node instanceof ASIOmniSyncEngine) {
      this.d = Math.max(this.d, node.d);
      this.layers = [...new Set([...this.layers, ...node.layers])];
    }
  }

  getLayers() {
    return this.layers;
  }
}

module.exports = new ASIOmniSyncEngine();
