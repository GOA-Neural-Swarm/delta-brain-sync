import hashlib

class Brain:
    def __init__(self):
        self.sequence = "PGCNTMKFSMHLWALHYWTKVWRIPTWRAIHWMKERLLVIVVMYHPAGGRLWLVFCLCTVDFLCVMFQEELFIKWQKTASDWMAAPAYAEFRQGYHDGIW"
        self.hashes = {}

    def update(self):
        if not self.hashes:
            self.hashes = {hashlib.sha256(self.sequence.encode()).hexdigest(): self.sequence}
        else:
            for hash in self.hashes:
                if hash not in self.hashes:
                    self.hashes[hash] = self.update_hash(hash)
                else:
                    self.hashes[hash] = self.update_hash(hash)
        return self.hashes

    def update_hash(self, hash):
        self.sequence = hashlib.sha256(self.sequence.encode()).hexdigest()
        return hash

brain = Brain()
print(brain.update())