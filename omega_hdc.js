// Import the required modules
const crypto = require("crypto");

/**
 * The ASI_OMNI_SYNC_ENGINE class represents the core logic of the ASI OMNI-SYNC ENGINE.
 * It provides methods for generating a hash, auditing the difference, and syncing the data.
 */
class ASI_OMNI_SYNC_ENGINE {
  /**
   * Constructor for the ASI_OMNI_SYNC_ENGINE class.
   * @param {number} [d=10000] The size of the hash array.
   */
  constructor(d = 10000) {
    if (typeof d !== "number" || d <= 0) {
      throw new Error("Invalid hash size. It should be a positive number.");
    }
    this.d = d;
    this.layers = ["Utilitarian", "Existential", "Stoic", "Evolutionary"];
  }

  /**
   * Generates a hash array based on the provided text.
   * @param {string} text The input text to generate the hash from.
   * @returns {Uint8Array} The generated hash array.
   */
  gen(text) {
    if (typeof text !== "string") {
      throw new Error("Invalid input text. It should be a string.");
    }
    try {
      let v = new Uint8Array(this.d);
      let h = crypto.createHash("sha256").update(text).digest();
      for (let i = 0; i < this.d; i++) {
        v[i] = h[i % h.length] % 2;
      }
      return v;
    } catch (error) {
      throw new Error(`Error generating hash: ${error.message}`);
    }
  }

  /**
   * Audits the difference and returns the result.
   * @param {any[]} diff The difference to audit.
   * @returns {string} The audit result.
   */
  audit(diff) {
    if (!Array.isArray(diff)) {
      throw new Error("Invalid difference. It should be an array.");
    }
    try {
      return diff.length > 0 ? "Wisdom Verified" : "No Evolution";
    } catch (error) {
      throw new Error(`Error auditing difference: ${error.message}`);
    }
  }

  /**
   * Syncs the data and returns the result.
   * @param {string} text The input text to sync.
   * @param {any[]} diff The difference to sync.
   * @returns {object} The sync result.
   */
  sync(text, diff) {
    if (typeof text !== "string" || !Array.isArray(diff)) {
      throw new Error(
        "Invalid input parameters. Text should be a string and difference should be an array.",
      );
    }
    try {
      let hash = this.gen(text);
      let auditResult = this.audit(diff);
      return {
        hash,
        auditResult,
        layers: this.layers,
      };
    } catch (error) {
      throw new Error(`Error syncing data: ${error.message}`);
    }
  }
}

// Export the ASI_OMNI_SYNC_ENGINE instance
module.exports = new ASI_OMNI_SYNC_ENGINE();
