if __name__ == "__main__":
    brain = Brain()
    brain.train(np.random.rand(10000), np.random.rand(10000))
    print(brain.think(np.random.rand(10000)))