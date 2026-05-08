# High-level abstraction for evolved services
from evolved_module import EvolvingClassifier, EvolvingRegressor
from util_module import LoggingUtility, ServiceMonitor
from data_module import DataValidator, FeatureExtractor, NewDataGenerator
from typing import List, Union
import time


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
        self.utilitarian_tracker = {
            "classifier_update_count": 0,
            "regressor_update_count": 0,
        }

    def handle_inference(
        self, feature_vector: List[float], inference_type: str
    ) -> Union[float, int, None]:
        """
        Handle inference for the given feature vector.

        Args:
        feature_vector (list): The input feature vector.
        inference_type (str): The type of inference (classification or regression).

        Returns:
        prediction: The predicted output.
        """
        try:
            # Validate input data
            if not self.validator.validate(feature_vector):
                self.logger.log_error("Invalid input data")
                return None

            # Extract features
            feature_vector = self.extractor.extract(feature_vector)

            # Monitor service performance
            start_time = time.time()
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
            end_time = time.time()
            self.monitor.stop_timer()
            self.monitor.log_performance(end_time - start_time)

            # Existential logic: track the number of inferences made
            if inference_type == "classification":
                self.utilitarian_tracker["classifier_update_count"] += 1
            elif inference_type == "regression":
                self.utilitarian_tracker["regressor_update_count"] += 1

            # Stoic logic: evaluate the performance of the services
            if (
                self.utilitarian_tracker["classifier_update_count"] > 100
                or self.utilitarian_tracker["regressor_update_count"] > 100
            ):
                self.logger.log_info(
                    f"Services have made over 100 inferences. Evaluating performance..."
                )
                self.monitor.evaluate_performance()

            return prediction
        except Exception as e:
            self.logger.log_error(f"Error occurred: {str(e)}")
            return None

    def evolve_services(
        self, num_iterations: int = 1, new_data_batch_size: int = 100
    ) -> None:
        """
        Evolve services by retraining the classifier and regressor with new data.

        Args:
        num_iterations (int): The number of evolution iterations. Defaults to 1.
        new_data_batch_size (int): The batch size of new data generated for each iteration. Defaults to 100.
        """
        try:
            for i in range(num_iterations):
                # Fetch new data
                new_data = self.new_data_generator.generate_new_data(
                    new_data_batch_size
                )

                # Validate new data
                if not self.validator.validate(new_data):
                    self.logger.log_error("Invalid new data")
                    return

                # Update classifier and regressor
                self.classifier.update(new_data)
                self.regressor.update(new_data)

                # Log evolution result
                self.logger.log_info(
                    f"Services evolved successfully (Iteration {i+1}/{num_iterations})"
                )

                # Evolutionary logic: track the number of updates made to the services
                self.utilitarian_tracker["classifier_update_count"] += 1
                self.utilitarian_tracker["regressor_update_count"] += 1
        except Exception as e:
            self.logger.log_error(f"Error occurred: {str(e)}")

    def start_app(self) -> None:
        """
        Start the EvolvedApp instance.
        """
        print(
            f"App Ready. Initial Inference: {self.handle_inference([1,2,3,4,5], 'classification')}"
        )
        print(
            f"App Ready. Initial Inference: {self.handle_inference([1,2,3,4,5], 'regression')}"
        )
        self.evolve_services(num_iterations=5, new_data_batch_size=500)


if __name__ == "__main__":
    app = EvolvedApp()
    app.start_app()
