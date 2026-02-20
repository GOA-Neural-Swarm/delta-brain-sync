import re
import string

class TelefoxX:
    def __init__(self):
        self.NeonDNA = "MCICPWTDGTEMYGTNRGHTFVSQPCGGHTSTVAHIYFFKVAERDGTIHGTTGCCTHPGPGLWCRRQQVVNFWFIHHDSIYAINCNTQCDYAAGHITRAGTCKTFNSDHGSVNCQTPIEGALAMFTKCRDPFYKSASTKHDEQIFTNNFD"
        self.Brain = ""

    def DNA_to_Brain(self):
        for i in range(len(self.NeonDNA)):
            if self.NeonDNA[i] in string.ascii_letters:
                self.Brain += chr(int(self.NeonDNA[i], 36))
            else:
                self.Brain += self.NeonDNA[i]

    def Upgrade_Brain(self):
        self.Brain = re.sub(r"[^A-Za-z0-9]+", "", self.Brain)
        self.Brain = re.sub(r"(\w)(\1{2,})", lambda m: m.group(1) + m.group(0).count(m.group(1)) // 2 * m.group(1), self.Brain)
        self.Brain = self.Brain.upper()

    def Run_Brain(self):
        print("TelefoxX Brain:")
        print(self.Brain)

telefoxx = TelefoxX()
telefoxx.DNA_to_Brain()
telefoxx.Upgrade_Brain()
telefoxx.Run_Brain()