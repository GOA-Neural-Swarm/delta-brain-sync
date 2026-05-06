import os

class HyperDimensionalLogger:
    def __init__(self):
        self.log_file = "hyper_dimensional_log.txt"
        self.utilitarian_metric = 0

    def log(self, message):
        if self.stoic_filter(message):
            with open(self.log_file, "a") as file:
                file.write(message + "\n")
            self.utilitarian_metric += 1

    def exists(self, path):
        return os.path.exists(path)

    def stoic_filter(self, message):
        if "error" in message.lower():
            return False
        return True

    def evolutionary_additive(self, new_message):
        if self.stoic_filter(new_message):
            with open(self.log_file, "r") as file:
                existing_lines = file.readlines()
            with open(self.log_file, "w") as file:
                file.write(new_message + "\n")
                file.writelines(existing_lines)
            self.utilitarian_metric += 1
            return True
        return False

def main():
    logger = HyperDimensionalLogger()
    print(logger.exists("/"))
    logger.log("Initial message")
    print(logger.utilitarian_metric)
    logger.evolutionary_additive("New additive message")
    print(logger.utilitarian_metric)

if __name__ == "__main__":
    main()