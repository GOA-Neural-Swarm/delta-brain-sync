// Merged and optimized configuration object
const CONFIG = {
  /**
   * Path to the instruction file
   * @type {string}
   */
  INSTRUCTION_FILE: "instruction.json",

  /**
   * Default command to execute
   * @type {string}
   */
  DEFAULT_COMMAND: "NORMAL_GROWTH",

  /**
   * Available modes with their respective growth rates
   * @type {Object<string, number>}
   */
  MODES: {
    /**
     * Hyper expansion mode with a growth rate of 50
     */
    HYPER_EXPANSION: 50,

    /**
     * Normal growth mode with a growth rate of 5
     */
    NORMAL_GROWTH: 5,

    /**
     * Stealth lockdown mode with a growth rate of 1
     */
    STEALTH_LOCKDOWN: 1,
  },

  /**
   * Maximum number of retries
   * @type {number}
   */
  RETRY_LIMIT: 3,

  /**
   * Timeout in milliseconds
   * @type {number}
   */
  TIMEOUT_MS: 15000,

  /**
   * Firebase service account credentials
   * @type {string}
   */
  FIREBASE_SERVICE_ACCOUNT: process.env.FIREBASE_SERVICE_ACCOUNT,

  /**
   * Neon database URL
   * @type {string}
   */
  NEON_DATABASE_URL: process.env.NEON_DATABASE_URL,
};

// Export the configuration object
module.exports = CONFIG;
