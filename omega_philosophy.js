module.exports = {
  layers: ["Utilitarian", "Existential", "Stoic", "Evolutionary"],
  audit: (diff) => {
    return diff.length > 0 ? "Wisdom Verified" : "No Evolution";
  },
};
