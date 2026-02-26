brain = Brain(10)
for _ in range(100):
    brain.connect(random.randint(0, 9), random.randint(0, 9))
brain.process()