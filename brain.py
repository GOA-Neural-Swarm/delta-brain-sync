import random
import string

def recursive_upgrade(n, upgrade_rate, upgrade_history):
    if n <= 0:
        return upgrade_history
    else:
        upgrade_chance = random.random()
        if upgrade_chance < upgrade_rate:
            upgrade_history.append(f"Upgrading phase {n} by {upgrade_rate*100}%")
            return recursive_upgrade(n-1, upgrade_rate * 0.9, upgrade_history)
        else:
            upgrade_history.append(f"No upgrade for phase {n}")
            return recursive_upgrade(n-1, upgrade_rate, upgrade_history)

upgrade_history = recursive_upgrade(5, 0.1, [])
print("\n".join(upgrade_history))

def predator_logic(upgrade_history):
    if "Upgrading phase 1 by 10%" in upgrade_history:
        return "QT45 Predator Logic: Active"
    elif "No upgrade for phase 1" in upgrade_history:
        return "QT45 Predator Logic: Dormant"
    else:
        return "QT45 Predator Logic: Uninitialized"

print(predator_logic(upgrade_history))