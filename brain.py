if __name__ == "__main__":
    brain = OptimizedBrain()
    inputs = np.random.rand(1000)
    targets = np.random.rand(1000)
    for _ in range(1000):
        brain.learn(inputs, targets)
    outputs = brain.process(inputs)
    print(outputs)