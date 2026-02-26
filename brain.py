if __name__ == '__main__':
    brain = Brain([[2], [3]])
    inputs = np.array([[1, 2, 3, 4]])
    targets = np.array([[1, 2]])
    brain.train(inputs, targets, 0.1)