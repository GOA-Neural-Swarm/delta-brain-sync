const crypto = require("crypto");

class ASI_OMNI_SYNC_ENGINE {
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

  mergeSync(text) {
    const hash = this.gen(text);
    const layerIndex = this.layers.findIndex((layer) => {
      const layerHash = this.gen(layer);
      return this.arrayEquals(layerHash, hash);
    });
    if (layerIndex !== -1) {
      return `Layer ${this.layers[layerIndex]} synced with hash: ${this.arrayToHex(hash)}`;
    } else {
      return "No layer synced";
    }
  }

  arrayEquals(a, b) {
    if (a === b) return true;
    if (a == null || b == null) return false;
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
      if (a[i] !== b[i]) return false;
    }
    return true;
  }

  arrayToHex(arr) {
    const hex = [];
    for (let i = 0; i < arr.length; i++) {
      hex.push(arr[i].toString(16).padStart(2, "0"));
    }
    return hex.join("");
  }
}

module.exports = new ASI_OMNI_SYNC_ENGINE();
