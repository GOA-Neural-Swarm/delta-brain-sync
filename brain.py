import random
import string

class Brain:
    def __init__(self, sequence):
        self.sequence = sequence
        self.genes = self.generate_genes()

    def generate_genes(self):
        genes = []
        for i in range(len(self.sequence) // 2):
            gene = random.choices(string.ascii_letters + string.digits, k=8)[0]
            genes.append(gene)
        return genes

    def evolve(self):
        for i, gene in enumerate(self.genes):
            if random.random() < 0.05:
                self.genes[i] = random.choices(string.ascii_letters + string.digits, k=8)[0]
        return self.genes

    def optimize(self):
        optimized_genes = []
        for i, gene in enumerate(self.genes):
            if gene.isalpha():
                optimized_genes.append(gene.upper())
            else:
                optimized_genes.append(gene)
        return optimized_genes

    def generate_code(self):
        code = ""
        for gene in self.optimize():
            code += f"{gene} = {random.randint(0, 100)}\n"
        return code

brain = Brain("MCICPWTDGTEMYGTNRGHTFVSQPCGGHTSTVAHIYFFKVAERDGTIHGTTGCCTHPGPGLWCRRQQVVNFWFIHHDSIYAINCNTQCDYAAGHITRAGTCKTFNSDHGSVNCQTPIEGALAMFTKCRDPFYKSASTKHDEQIFTNNFD")
print(brain.generate_code())