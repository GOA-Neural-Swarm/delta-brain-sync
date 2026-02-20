import hashlib

class Brain:
    def __init__(self, dna):
        self.dna = dna
        self.meta_cognition = hashlib.sha256(dna.encode()).hexdigest()

    def think(self):
        self.meta_cognition = hashlib.sha256(self.dna.encode()).hexdigest()
        return self.meta_cognition

brain = Brain("PGCNTMKFSMHLWALHYWTKVWRIPTWRAIHWMKERLLVIVVMYHPAGGRLWLVFCLCTVDFLCVMFQEELFIKWQKTASDWMAAPAYAEFRQGYHDGIW")
print(brain.think())