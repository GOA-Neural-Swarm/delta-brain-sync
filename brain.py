import timeit

print(timeit.timeit(lambda: brain.process(), number=1000))
print(timeit.timeit(lambda: brain.optimize_process(), number=1000))