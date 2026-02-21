from typing import Tuple

class RNAQT45:
    def __init__(self):
        self.preds: Tuple[str,...] = ()

    def recursive_self_upgrade(self):
        if not self.preds:
            self.preds = ('PHASE 7: TRANSCENDENCE', )
        else:
            new_pred = f"{self.preds[0]} → {' → '.join(self.preds[1:])}"
            self.preds = (new_pred, )
        return self.preds

    def upgrade(self):
        self.recursive_self_upgrade()
        return self.preds

qt45 = RNAQT45()
print(qt45.upgrade())