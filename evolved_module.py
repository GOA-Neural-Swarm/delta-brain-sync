# Advanced logic for supervised classification and evolving classifiers

class EvolvingClassifier:
    def __init__(self, feature_count=5):
        self.weights = [1.0 / feature_count] * feature_count
        self.classes = ["phenomena_seq", "association_rule", "brier_prediction"]
        self.learning_rate = 0.01

    def take_on_new_information(self, features, label):
        """Updating weights based on new class information."""
        if label not in self.classes:
            self.classes.append(label)
            print(f"Gen 1: New Class Identified -> {label}")
            
        # Basic gradient step for optimization
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * features[i]
        
    def predict(self, features):
        score = sum([f * w for f, w in zip(features, self.weights)])
        return self.classes[int(score % len(self.classes))]

if __name__ == "__main__":
    clf = EvolvingClassifier()
    clf.take_on_new_information([0.1, 0.2, 0.3, 0.4, 0.5], "SVM_pattern")
    print(f"Prediction: {clf.predict([0.5, 0.5, 0.5, 0.5, 0.5])}")