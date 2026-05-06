# High-level abstraction for evolved services
from evolved_module import EvolvingClassifier, EvolvingRegressor
from util_module import LoggingUtility, ServiceMonitor
from data_module import DataValidator, FeatureExtractor, NewDataGenerator


class EvolvedApp:
    def __init__(self):
        """
        Initialize the EvolvedApp instance.

        Attributes:
        classifier (EvolvingClassifier): The evolving classifier instance.
        regressor (EvolvingRegressor): The evolving regressor instance.
        logger (LoggingUtility): The logging utility instance.
        monitor (ServiceMonitor): The service monitor instance.
        validator (DataValidator): The data validator instance.
        extractor (FeatureExtractor): The feature extractor instance.
        new_data_generator (NewDataGenerator): The new data generator instance.
        """
        self.classifier = EvolvingClassifier()
        self.regressor = EvolvingRegressor()
        self.logger = LoggingUtility()
        self.monitor = ServiceMonitor()
        self.validator = DataValidator()
        self.extractor = FeatureExtractor()
        self.new_data_generator = NewDataGenerator()

    def handle_inference(self, feature_vector, inference_type):
        """
        Handle inference for the given feature vector.

        Args:
        feature_vector (list): The input feature vector.
        inference_type (str): The type of inference (classification or regression).

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

        if inference_type == "classification":
            # Perform inference using classifier
            prediction = self.classifier.predict(feature_vector)
        elif inference_type == "regression":
            # Perform inference using regressor
            prediction = self.regressor.predict(feature_vector)
        else:
            self.logger.log_error("Invalid inference type")
            return None

        # Log inference result
        self.logger.log_info(f"Inference result: {prediction}")

        # Monitor service performance
        self.monitor.stop_timer()
        self.monitor.log_performance()

        return prediction

    def evolve_services(self):
        """
        Evolve services by retraining the classifier and regressor with new data.
        """
        # Fetch new data
        new_data = self.new_data_generator.generate_new_data()

        # Validate new data
        if not self.validator.validate(new_data):
            self.logger.log_error("Invalid new data")
            return

        # Update classifier and regressor
        self.classifier.update(new_data)
        self.regressor.update(new_data)

        # Log evolution result
        self.logger.log_info("Services evolved successfully")


if __name__ == "__main__":
    app = EvolvedApp()
    print(
        f"App Ready. Initial Inference: {app.handle_inference([1,2,3,4,5], 'classification')}"
    )
    print(
        f"App Ready. Initial Inference: {app.handle_inference([1,2,3,4,5], 'regression')}"
    )
    app.evolve_services()
