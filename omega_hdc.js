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

  mergePhilosophy(text) {
    const hash = this.gen(text);
    const layerIndex = this.layers.reduce(
      (acc, layer, index) => {
        const layerHash = this.gen(layer);
        const similarity = this.similarity(hash, layerHash);
        return similarity > acc.similarity ? { index, similarity } : acc;
      },
      { index: -1, similarity: 0 },
    );
    return this.layers[layerIndex.index];
  }

  similarity(a, b) {
    let similarity = 0;
    for (let i = 0; i < a.length; i++) {
      if (a[i] === b[i]) similarity++;
    }
    return similarity / a.length;
  }
}

module.exports = new HDC();
