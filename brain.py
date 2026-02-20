brain = Brain()
for _ in range(10):
    node = NeuroNode(_)
    brain.add_node(node)

brain.process()
output = brain.get_output()
print(output)