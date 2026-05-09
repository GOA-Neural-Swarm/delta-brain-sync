from evolved_module import EvolvingClassifier, EvolvingRegressor
from util_module import LoggingUtility, ServiceMonitor
from data_module import DataValidator, FeatureExtractor, NewDataGenerator
from typing import List, Union
import time

class EvolvedApp:
    def __init__(self):
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
        self.stoic_service_tracker = {
            "classifier_service_performance": 0.0,
            "regressor_service_performance": 0.0,
        }

    def handle_inference(self, feature_vector: List[float], inference_type: str) -> Union[float, int, None]:
        try:
            if not self.validator.validate(feature_vector):
                self.logger.log_error("Invalid input data")
                return None

            feature_vector = self.extractor.extract(feature_vector)
            start_time = time.time()
            self.monitor.start_timer()

            if inference_type == "classification":
                prediction = self.classifier.predict(feature_vector)
                self.stoic_service_tracker["classifier_service_performance"] = self.monitor.evaluate_service_performance(start_time)
            elif inference_type == "regression":
                prediction = self.regressor.predict(feature_vector)
                self.stoic_service_tracker["regressor_service_performance"] = self.monitor.evaluate_service_performance(start_time)
            else:
                self.logger.log_error("Invalid inference type")
                return None

            self.logger.log_info(f"Inference result: {prediction}")
            end_time = time.time()
            self.monitor.stop_timer()
            self.monitor.log_performance(end_time - start_time)

            if inference_type == "classification":
                self.utilitarian_tracker["classifier_update_count"] += 1
            elif inference_type == "regression":
                self.utilitarian_tracker["regressor_update_count"] += 1

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

    def evolve_services(self, num_iterations: int = 1, new_data_batch_size: int = 100) -> None:
        try:
            for i in range(num_iterations):
                new_data = self.new_data_generator.generate_new_data(new_data_batch_size)

                if not self.validator.validate(new_data):
                    self.logger.log_error("Invalid new data")
                    return

                self.classifier.update(new_data)
                self.regressor.update(new_data)

                self.logger.log_info(
                    f"Services evolved successfully (Iteration {i+1}/{num_iterations})"
                )

                self.utilitarian_tracker["classifier_update_count"] += 1
                self.utilitarian_tracker["regressor_update_count"] += 1
        except Exception as e:
            self.logger.log_error(f"Error occurred: {str(e)}")

    def start_app(self) -> None:
        print(f"App Ready. Initial Inference: {self.handle_inference([1,2,3,4,5], 'classification')}")
        print(f"App Ready. Initial Inference: {self.handle_inference([1,2,3,4,5], 'regression')}")
        self.evolve_services(num_iterations=5, new_data_batch_size=500)

if __name__ == "__main__":
    app = EvolvedApp()
    app.start_app()