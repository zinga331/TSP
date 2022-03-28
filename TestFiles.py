import numpy as np
from TSPSolver import TSPSolver

class Scenario:
    def __init__(self, cities):
        self.cities = cities

    def getCities(self):
        return self.cities

class City:
    def __init__(self, index, name):
        self._index = index
        self._name = name

    def costTo(self, other_city):
        if self._index == other_city._index:
            return np.inf
        elif self._name == 'A' and other_city._name == 'B':
            return 7
        elif self._name == 'A' and other_city._name == 'C':
            return 3
        elif self._name == 'A' and other_city._name == 'D':
            return 12

        elif self._name == 'B' and other_city._name == 'A':
            return 3
        elif self._name == 'B' and other_city._name == 'C':
            return 6
        elif self._name == 'B' and other_city._name == 'D':
            return 14

        elif self._name == 'C' and other_city._name == 'A':
            return 5
        elif self._name == 'C' and other_city._name == 'B':
            return 8
        elif self._name == 'C' and other_city._name == 'D':
            return 6
        elif self._name == 'D' and other_city._name == 'A':
            return 9
        elif self._name == 'D' and other_city._name == 'B':
            return 3
        elif self._name == 'D' and other_city._name == 'C':
            return 5
cities = [City(0, 'A'), City(1, 'B'), City(2, 'C'), City(3, 'D')]
scenario = Scenario(cities)
tsp_solver = TSPSolver(None)
tsp_solver.setupWithScenario(scenario)
solution = tsp_solver.branchAndBound()
print(solution['cost'])
# print([city._name for city in solution['soln'].route])
# print(solution['soln'].cost)