#!/usr/bin/python3
import copy
import math
import queue
import numpy

from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT6':
    from PyQt6.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
import heapq
import itertools


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    ''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution,
		time spent to find solution, number of permutations tried during search, the
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    def greedy(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()

        while not foundTour and time.time() - start_time < time_allowance:
            route = []
            curr_city_index = np.random.randint(0, ncities)
            # Add the starting city to the route, and show that we HAVE visited it.
            route.append(cities[curr_city_index])
            for num_visited in range(ncities):
                new_index = None
                min_edge = math.inf
                cities[curr_city_index].visited = True  # We will visit this city, set it to visited.
                for i in range(ncities):
                    new_dist = cities[curr_city_index].costTo(cities[i])
                    if (not cities[i].visited) and min_edge > new_dist:
                        min_edge = new_dist
                        new_index = i
                if min_edge == math.inf:
                    break

                curr_city_index = new_index
                if curr_city_index is not None:
                    route.append(cities[curr_city_index])
            bssf = TSPSolution(route)
            if len(route) < len(cities):
                bssf.cost = math.inf
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    def branchAndBound(self, time_allowance=60.0):
        start_time = time.time()
        priority_queue = queue.PriorityQueue()
        cities = self._scenario.getCities()
        ncities = len(cities)
        count = 0
        max_states = 1
        total_states = 1  # counting for the start state.
        pruned_states = 0

        bssf = self.greedy(time_allowance / 100)['soln']
        table, lower_bound = self.compute_minimum_matrix()
        start_state = State(table, lower_bound, 1,
                            0)  # Hard-code the depth of the tree to be one, since this is the first
        # state.
        start_state.route.append(start_state.last)
        for row in range(ncities):
            start_state.cost_table[row][0] = math.inf
        for destination in range(ncities):
            new_state = self.expand(start_state, start_state.last, destination)
            total_states += 1
            new_state.route.append(destination)
            if new_state.lower_bound < bssf.cost:
                priority_queue.put(new_state)
                max_states = max(max_states, len(priority_queue.queue))
            else:
                pruned_states += 1
        # Todo Expand the substates properly, making sure not to expand something already expanded.
        # Enter the while loop that works the problem until the queue is empty.
        while len(priority_queue.queue) > 0 and time.time() - start_time < time_allowance:
            curr_state = priority_queue.get()
            if curr_state.lower_bound < bssf.cost:
                for destination in range(ncities):
                    new_state = self.expand(curr_state, curr_state.last, destination)
                    total_states += 1
                    new_state.route.append(destination)
                    if new_state.lower_bound < bssf.cost:
                        if len(new_state.route) == ncities:
                            closing_cost = cities[new_state.route[-1]].costTo(cities[new_state.route[0]])
                            if closing_cost != math.inf:
                                new_route = self.make_route(new_state.route, cities)
                                pending_solution = TSPSolution(new_route)
                                if bssf.cost > pending_solution.cost:
                                    bssf = pending_solution
                                    count += 1
                        else:
                            priority_queue.put(new_state)
                    else:
                        pruned_states += 1
            else:
                pruned_states += 1
            max_states = max(max_states, len(priority_queue.queue))
        pruned_states += len(priority_queue.queue)
        end_time = time.time()
        # TSPSolution(solution_cities)
        results = {}
        print(len(priority_queue.queue))
        results['cost'] = bssf.cost
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = max_states
        results['total'] = total_states
        results['pruned'] = pruned_states
        return results

    def expand(self, curr_state, node_row, node_column):
        if curr_state.cost_table[node_row][node_column] == math.inf:
            return State(curr_state.cost_table, math.inf, 1, 1)
        table = copy.deepcopy(curr_state)
        table.lower_bound += curr_state.cost_table[node_row][node_column]
        for column in range(table.cost_table.shape[0]):
            table.cost_table[node_row][column] = math.inf
        for row in range(table.cost_table.shape[0]):
            table.cost_table[row][node_column] = math.inf
        table.lower_bound += self.reduce_matrix(table.cost_table)
        table.depth += 1
        table.last = node_column
        # update the cost matrix of the row and column that match the expansion to infinity.
        return table

    # Previous iterations of my program had each state hold onto a list of city object. Holding the indices sped things up a bunch.
    def make_route(self, city_indices, all_cities):
        route = []
        for i in range(len(city_indices)):
            route.append(all_cities[city_indices[i]])
        return route

    def compute_minimum_matrix(self):
        cities = self._scenario.getCities()
        ncities = len(cities)
        # use a numpy array instead of an nd array. That'll
        min_table = np.zeros((ncities, ncities))
        for i in range(ncities):
            for j in range(ncities):
                min_table[i, j] = cities[i].costTo(cities[j])
        # add all the costs for city i to a matrix.
        lower_bound = self.reduce_matrix(min_table)

        return min_table, lower_bound

    def reduce_matrix(self, min_table):
        min_cost = 0
        for row in range(min_table.shape[0]):
            min_of_row = numpy.amin(min_table[[row], :])
            if min_of_row != math.inf:
                min_table[row] -= min_of_row
                min_cost += min_of_row
        for column in range(min_table.shape[1]):
            min_of_column = numpy.amin(min_table[:, column])
            if min_of_column != math.inf:
                min_table[:, column] -= min_of_column
                min_cost += min_of_column
        return min_cost

    # consider creating a TSP class node, getPriority, and set the lambda as node.getPriority.
    # keep track of your current cost matrix within this node as well.

    ''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found during search, the
		best solution found.  You may use the other three field however you like.
		algorithm</returns>
	'''

    def fancy(self, time_allowance=60.0):
        pass


class State:
    def __init__(self, cost_table, lower_bound, depth, last):
        self.route = []
        self.cost_table = cost_table
        self.lower_bound = lower_bound
        self.depth = depth
        self.last = last
        self.ncities = len(cost_table[0])

    def __lt__(self, other):  # return true if other's priority is less than self's.
        priority_self = self.get_priority()
        priority_other = other.get_priority()
        if priority_other > priority_self:  # Todo see if this backwards understanding helps.
            return True
        else:
            return False

    # Adjusting this function allows modifies which state it expands first, hopefully to calculate new solutions and
    # allow other states to be pruned sooner.
    def get_priority(self):
        percentage = self.depth / self.ncities
        if percentage >= .35:
            discount = self.depth = percentage / 5
            priority = self.lower_bound - (self.lower_bound * discount)
        else:
            priority = self.lower_bound
        return priority
    # if len(self.route) < (self.ncities - (self.ncities / 10)):
    #     priority = self.lower_bound / (reduce * .5)
    # else:
    #     priority = self.lower_bound / (reduce * .1)
    # return priority
