import timeit

def benchmark_brain(benchmark_brain_instance):
    for _ in range(1000):
        benchmark_brain_instance.propagate()

benchmark_brain_instance = Brain(1000)
start_time = timeit.default_timer()
benchmark_brain(benchmark_brain_instance)
end_time = timeit.default_timer()
print(f"Propagation time: {end_time - start_time:.6f} seconds")