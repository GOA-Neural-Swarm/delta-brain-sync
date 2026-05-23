import telemetry_bridge
import os


def calculate_utility(evolving_system):
    """
    Calculate the overall utility of an evolving system.
    """
    return evolving_system.get.utility()


class Utility:

    def __init__(self):
        self.utility_values = []

    def add_utility(self, value):
        self.utility_values.append(value)

    def get(self):
        return self.utility_values[-1]


class StoicFunction:

    def __init__(self):
        self.evolving_system = EvolvingSystem()

    def stoic_function(self):
        print("Stoic function added")
        self.evolving_system.add_component(self)


class EvolvingSystem:

    def __init__(self):
        self.components = []
        self.utility = Utility()

    def add_component(self, component):
        self.components.append(component)
        self.update_utility()

    def update_utility(self):
        self.utility.add_utility(len(self.components))


evolving_system = EvolvingSystem()
stoic_function = StoicFunction()
stoic_function.stoic_function()
utility = evolving_system.utility
print("Current utility:", utility.get())


class NewComponent:
    pass


evolving_system.add_component(NewComponent())
print("Updated utility:", evolving_system.utility.get())
