import time
start_time = time.time()

brain = Brain()
brain.optimize(np.array([[0.5]]), np.array([[0.8]]), iterations=10000)
print("Optimized brain: ", brain.predict(np.array([[0.5]])))

end_time = time.time()
print("Time taken: ", end_time - start_time)
