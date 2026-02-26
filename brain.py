def main():
    brain = Brain()
    brain.add_layer(Layer(np.array([[1, 2]]), np.array([1])))
    brain.add_optimizer(Optimizer(0.01))

    inputs = np.array([[1, 2], [3, 4]])
    targets = np.array([1, 2])

    brain.train(inputs, targets)

if __name__ == "__main__":
    main()