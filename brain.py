import random
import string

class Brain:
    def __init__(self):
        self.memory = []
        self.max_chunk_size = 10
        self.min_chunk_size = 1

    def learn(self, input_data):
        self.memory.append(input_data)

    def recall(self):
        if len(self.memory) > 0:
            return self.memory[-1]
        else:
            return None

    def evolve(self):
        new_memory = []
        for memory_chunk in self.memory:
            if random.random() < 0.5:
                new_memory.append(memory_chunk)
            else:
                chunk_size = random.randint(self.min_chunk_size, self.max_chunk_size)
                chunk = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(chunk_size))
                new_memory.append(chunk)
        self.memory = new_memory

    def generate_random_chunk(self):
        chunk_size = random.randint(self.min_chunk_size, self.max_chunk_size)
        chunk = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(chunk_size))
        return chunk

brain = Brain()
brain.learn("Initial Knowledge")
brain.learn("More Information")
brain.learn("New Insights")

print(brain.recall())
brain.evolve()
print(brain.recall())