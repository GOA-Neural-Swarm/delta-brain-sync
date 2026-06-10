const crypto = require("crypto");

/**
 * ASI_OMNI_SYNC_ENGINE class representing the core logic for generating hashes,
 * calculating differences, and auditing the sync process.
 */
class ASI_OMNI_SYNC_ENGINE {
  /**
   * Constructor for initializing the ASI_OMNI_SYNC_ENGINE.
   * @param {number} d - The dimension or size of the hash array. Defaults to 10000.
   */
  constructor(d = 10000) {
    this.d = d;
    this.layers = ["Utilitarian", "Existential", "Stoic", "Evolutionary"];
    this.storedHash = new Uint8Array(this.d); // Initialize storedHash
  }

  /**
   * Generates a hash from the given text and returns it as a Uint8Array.
   * @param {string} text - The text to generate the hash from.
   * @returns {Uint8Array} The generated hash.
   */
  gen(text) {
    const hash = crypto.createHash("sha256").update(text).digest();
    const v = new Uint8Array(this.d);
    for (let i = 0; i < this.d; i++) {
      v[i] = hash[i % hash.length] % 2;
    }
    return v;
  }

  /**
   * Audits the difference array and returns a verification message.
   * @param {number[]} diff - The array of differences.
   * @returns {string} The audit result message.
   */
  audit(diff) {
    return diff.length > 0 ? "Wisdom Verified" : "No Evolution";
  }

  /**
   * Syncs a node based on the provided text and returns the audit result.
   * @param {string} text - The text to sync the node with.
   * @returns {string} The result of the audit.
   */
  syncNode(text) {
    const hash = this.gen(text);
    const diff = this.calculateDiff(hash);
    return this.audit(diff);
  }

  /**
   * Calculates the difference between the generated hash and the stored hash.
   * @param {Uint8Array} hash - The generated hash to compare with the stored hash.
   * @returns {number[]} An array of indices where the hashes differ.
   */
  calculateDiff(hash) {
    const diff = [];
    for (let i = 0; i < hash.length; i++) {
      if (hash[i] !== this.storedHash[i]) {
        diff.push(i);
      }
    }
    return diff;
  }

  /**
   * Updates the stored hash.
   * @param {Uint8Array} newHash - The new hash to store.
   */
  updateStoredHash(newHash) {
    this.storedHash = newHash;
  }

  /**
   * Gets the currently stored hash.
   * @returns {Uint8Array} The stored hash.
   */
  getStoredHash() {
    return this.storedHash;
  }
}

module.exports = new ASI_OMNI_SYNC_ENGINE();
