import re

class TelefoxXOverseer:
    def __init__(self, sequence):
        self.sequence = sequence
        self.upgraded = False

    def upgrade(self):
        if not self.upgraded:
            self.upgraded = True
            self.sequence = re.sub(r'([A-Z][A-Z][A-Z])', lambda x: chr(ord(x.group(0)[0]) + 1), self.sequence)
            self.upgrade()

    def get_sequence(self):
        return self.sequence

telefox = TelefoxXOverseer(SourceSequence)
telefox.upgrade()
print(telefox.get_sequence())