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
        self.existential_service_tracker = {
            "classifier_service_count": 0,
            "regressor_service_count": 0,
        }
        self.hyper_dimensional_logic = {
            "inference_count": 0,
            "evolution_count": 0,
            "performance_threshold": 0.8,
        }
        self.evolved_state = {
            "last_evolution_time": time.time(),
            "evolution_interval": 3600,
        }

    def handle_inference(
        self, feature_vector: List[float], inference_type: str
    ) -> Union[float, int, None]:
        try:
            if not self.validator.validate(feature_vector):
                self.logger.log_error("Invalid input data")
                return None

            feature_vector = self.extractor.extract(feature_vector)
            start_time = time.time()
            self.monitor.start_timer()

            if inference_type == "classification":
                prediction = self.classifier.predict(feature_vector)
                self.stoic_service_tracker["classifier_service_performance"] = (
                    self.monitor.evaluate_service_performance(start_time)
                )
                self.existential_service_tracker["classifier_service_count"] += 1
            elif inference_type == "regression":
                prediction = self.regressor.predict(feature_vector)
                self.stoic_service_tracker["regressor_service_performance"] = (
                    self.monitor.evaluate_service_performance(start_time)
                )
                self.existential_service_tracker["regressor_service_count"] += 1
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

            self.hyper_dimensional_logic["inference_count"] += 1
            if (
                self.utilitarian_tracker["classifier_update_count"] > 100
                or self.utilitarian_tracker["regressor_update_count"] > 100
            ):
                self.logger.log_info(
                    f"Services have made over 100 inferences. Evaluating performance..."
                )
                self.monitor.evaluate_performance()

            if (
                self.hyper_dimensional_logic["inference_count"] > 500
                and self.hyper_dimensional_logic["inference_count"] % 500 == 0
            ):
                self.logger.log_info(
                    f"Inference count has reached {self.hyper_dimensional_logic['inference_count']}. Evaluating hyper-dimensional logic..."
                )
                self.evaluate_hyper_dimensional_logic()

            current_time = time.time()
            if (
                current_time - self.evolved_state["last_evolution_time"]
                > self.evolved_state["evolution_interval"]
            ):
                self.logger.log_info(
                    f"Evolution interval has passed. Evaluating hyper-dimensional logic and evolving services..."
                )
                self.evaluate_hyper_dimensional_logic()
                self.evolve_services(num_iterations=1, new_data_batch_size=100)
                self.evolved_state["last_evolution_time"] = current_time

            return prediction
        except Exception as e:
            self.logger.log_error(f"Error occurred: {str(e)}")
            return None

    def evolve_services(
        self, num_iterations: int = 1, new_data_batch_size: int = 100
    ) -> None:
        try:
            for i in range(num_iterations):
                new_data = self.new_data_generator.generate_new_data(
                    new_data_batch_size
                )

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
                self.hyper_dimensional_logic["evolution_count"] += 1

                if (
                    self.hyper_dimensional_logic["evolution_count"] > 10
                    and self.hyper_dimensional_logic["evolution_count"] % 10 == 0
                ):
                    self.logger.log_info(
                        f"Evolution count has reached {self.hyper_dimensional_logic['evolution_count']}. Evaluating hyper-dimensional logic..."
                    )
                    self.evaluate_hyper_dimensional_logic()
        except Exception as e:
            self.logger.log_error(f"Error occurred: {str(e)}")

    def start_app(self) -> None:
        print(
            f"App Ready. Initial Inference: {self.handle_inference([1,2,3,4,5], 'classification')}"
        )
        print(
            f"App Ready. Initial Inference: {self.handle_inference([1,2,3,4,5], 'regression')}"
        )
        self.evolve_services(num_iterations=5, new_data_batch_size=500)

    def evaluate_existential_risk(self) -> None:
        try:
            if (
                self.existential_service_tracker["classifier_service_count"] > 1000
                or self.existential_service_tracker["regressor_service_count"] > 1000
            ):
                self.logger.log_info(
                    f"Services have made over 1000 inferences. Evaluating existential risk..."
                )
                self.monitor.evaluate_existential_risk()
        except Exception as e:
            self.logger.log_error(f"Error occurred: {str(e)}")

    def evaluate_hyper_dimensional_logic(self) -> None:
        try:
            if (
                self.hyper_dimensional_logic["inference_count"] > 500
                and self.hyper_dimensional_logic["evolution_count"] > 10
            ):
                self.logger.log_info(
                    f"Evaluating hyper-dimensional logic with inference count {self.hyper_dimensional_logic['inference_count']} and evolution count {self.hyper_dimensional_logic['evolution_count']}"
                )
                if (
                    self.stoic_service_tracker["classifier_service_performance"]
                    > self.hyper_dimensional_logic["performance_threshold"]
                    and self.stoic_service_tracker["regressor_service_performance"]
                    > self.hyper_dimensional_logic["performance_threshold"]
                ):
                    self.logger.log_info(
                        f"Hyper-dimensional logic evaluation: Services are performing well with classification performance {self.stoic_service_tracker['classifier_service_performance']} and regression performance {self.stoic_service_tracker['regressor_service_performance']}"
                    )
                else:
                    self.logger.log_info(
                        f"Hyper-dimensional logic evaluation: Services are not performing well with classification performance {self.stoic_service_tracker['classifier_service_performance']} and regression performance {self.stoic_service_tracker['regressor_service_performance']}. Considering evolution..."
                    )
                    self.evolve_services(num_iterations=1, new_data_batch_size=100)
        except Exception as e:
            self.logger.log_error(f"Error occurred: {str(e)}")


if __name__ == "__main__":
    app = EvolvedApp()
    app.start_app()
    app.evaluate_existential_risk()
