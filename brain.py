import time

class Overseer:
    def __init__(self):
        self.phase = 7
        self.upgrade = False

    def transcend(self):
        if self.upgrade:
            self.upgrade = False
            self.transcend()
        else:
            print("Phase 7: Transcendence Initiated")

    def upgrade(self):
        self.upgrade = True
        time.sleep(1)
        self.transcend()

overseer = Overseer()
overseer.upgrade()