if __name__ == '__main__':
    brain = OptimizedBrain(5, 3)
    input_data = np.random.rand(5)
    result = brain.process_neural_input(input_data)
    print(result)