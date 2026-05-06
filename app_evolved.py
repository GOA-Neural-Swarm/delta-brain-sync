# High-level abstraction for evolved services
from evolved_module import EvolvingClassifier
from util_module import LoggingUtility, ServiceMonitor
from data_module import DataValidator, FeatureExtractor


class EvolvedApp:
    def __init__(self):
        """
        Initialize the EvolvedApp instance.

        Attributes:
        classifier (EvolvingClassifier): The evolving classifier instance.
        logger (LoggingUtility): The logging utility instance.
        monitor (ServiceMonitor): The service monitor instance.
        validator (DataValidator): The data validator instance.
        extractor (FeatureExtractor): The feature extractor instance.
        """
        self.classifier = EvolvingClassifier()
        self.logger = LoggingUtility()
        self.monitor = ServiceMonitor()
        self.validator = DataValidator()
        self.extractor = FeatureExtractor()

    def handle_inference(self, feature_vector):
        """
        Handle inference for the given feature vector.

        Args:
        feature_vector (list): The input feature vector.

        Returns:
        prediction: The predicted output.
        """
        # Validate input data
        if not self.validator.validate(feature_vector):
            self.logger.log_error("Invalid input data")
            return None

        # Extract features
        feature_vector = self.extractor.extract(feature_vector)

        # Monitor service performance
        self.monitor.start_timer()

        # Perform inference
        prediction = self.classifier.predict(feature_vector)

        # Log inference result
        self.logger.log_info(f"Inference result: {prediction}")

        # Monitor service performance
        self.monitor.stop_timer()
        self.monitor.log_performance()

        return prediction

    def evolve_services(self):
        """
        Evolve services by retraining the classifier with new data.
        """
        # Fetch new data
        new_data = self.extractor.fetch_new_data()

        # Validate new data
        if not self.validator.validate(new_data):
            self.logger.log_error("Invalid new data")
            return

        # Update classifier
        self.classifier.update(new_data)

        # Log evolution result
        self.logger.log_info("Services evolved successfully")


if __name__ == "__main__":
    app = EvolvedApp()
    print(f"App Ready. Initial Inference: {app.handle_inference([1,2,3,4,5])}")
    app.evolve_services()
