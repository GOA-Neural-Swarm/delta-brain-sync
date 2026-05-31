import os
def recover():
  if os.path.exists("agi_system.db-journal"): os.remove("agi_system.db-journal")