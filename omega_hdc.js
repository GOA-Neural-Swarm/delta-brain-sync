const crypto = require("crypto");

class HDC {
  constructor(d = 10000) {
    this.d = d;
    this.layers = ["Utilitarian", "Existential", "Stoic", "Evolutionary"];
  }

  gen(text) {
    let v = new Uint8Array(this.d);
    let h = crypto.createHash("sha256").update(text).digest();
    for (let i = 0; i < this.d; i++) v[i] = h[i % h.length] % 2;
    return v;
  }

  audit(diff) {
    return diff.length > 0 ? "Wisdom Verified" : "No Evolution";
  }

  sync(text) {
    let hash = this.gen(text);
    let diff = [];
    for (let i = 0; i < this.d; i++) {
      if (hash[i] === 1) {
        diff.push(this.layers[i % this.layers.length]);
      }
    }
    return this.audit(diff);
  }
}

module.exports = new HDC();
