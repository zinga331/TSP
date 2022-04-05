#!/usr/bin/python3
import copy
import math
import queue
import numpy
import random

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
        num_cities_checked = 0
        city_starts = []

        while not foundTour and time.time() - start_time < time_allowance and len(city_starts) < ncities:
            for city in range(ncities):
                cities[city].visited = False
            route = []
            found_city = False
            while not found_city and len(city_starts) < ncities:
                num_cities_checked += 1
                rand_city = np.random.randint(0, ncities)
                if not city_starts.count(rand_city):
                    city_starts.append(rand_city)
                    curr_city_index = rand_city
                    found_city = True
            if len(city_starts) == ncities:
                break

            # curr_city_index = city_index
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
        count = 0  # count represents the total number of solutions found. If count ends as 0, only a greedy solution
        # was found.
        max_states = 1
        total_states = 1  # counting for the start state.
        pruned_states = 0

        # Operating under the assumption that most trials will be given 60 seconds, I don't want to spend more than 2
        # seconds to get an initial BSSF from the greed algorithm.
        bssf = self.greedy(time_allowance / 30)['soln']
        table, lower_bound = self.compute_minimum_matrix()
        start_state = State(table, lower_bound, 1, 0)  # Hard-code the depth of the tree to be one, since this is the
        # first state.
        start_state.route.append(start_state.last)
        # A valid cycle for TSP won't include an edge going to the first city until a solution, blanking out the
        # first column.
        for row in range(ncities):
            start_state.cost_table[row][0] = math.inf
        # Expand the first state into new states including an edge from city 0 to other cities.
        for destination in range(ncities):
            new_state = self.expand(start_state, start_state.last, destination)
            total_states += 1
            new_state.route.append(destination)
            if new_state.lower_bound < bssf.cost:
                priority_queue.put(new_state)
                max_states = max(max_states, len(priority_queue.queue))
            else:
                pruned_states += 1

        # Enter the while loop that works the problem until the queue is empty.
        while len(priority_queue.queue) > 0 and time.time() - start_time < time_allowance:
            curr_state = priority_queue.get()
            if curr_state.lower_bound < bssf.cost:
                for destination in range(ncities):
                    new_state = self.expand(curr_state, curr_state.last, destination)
                    total_states += 1  # Add to total states, even if it is pruned shortly after.
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
                    else:  # if the cost isn't better than bssf, prune it.
                        pruned_states += 1
            else:
                pruned_states += 1  # if the state no longer belongs on the queue, prune it.
            max_states = max(max_states, len(priority_queue.queue))
        # if time in the while loop runs out, count the remaining states as pruned.
        pruned_states += len(priority_queue.queue)
        end_time = time.time()
        # TSPSolution(solution_cities)
        results = {}  # create a new results object.
        print(len(priority_queue.queue))
        results['cost'] = bssf.cost
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = max_states
        results['total'] = total_states
        results['pruned'] = pruned_states
        return results

    # Given the cost between each cities, create the minimum matrix.
    def compute_minimum_matrix(self):  # Time O(n^2), Space O(n^2)
        # The cost between each city is calculated in both directions, to give an n by n matrix
        cities = self._scenario.getCities()
        ncities = len(cities)
        min_table = np.zeros((ncities, ncities))  # Create ncities by ncities matrix to hold the cost values.
        # nested loop of size n running n times.
        for i in range(ncities):
            for j in range(ncities):
                min_table[i, j] = cities[i].costTo(cities[j])
        # add all the costs for city i to a matrix.
        lower_bound = self.reduce_matrix(min_table)

        return min_table, lower_bound

    # min_cost doesn't include the lower bound so far, just the cost to minimize the matrix.
    def reduce_matrix(self, min_table):  # O(n^2)
        min_cost = 0
        # reduce each row, finding the min and subtracting the min from each row.
        for row in range(min_table.shape[0]):  # n^2 checks at constant time.
            min_of_row = numpy.amin(min_table[[row], :])
            if min_of_row != math.inf:
                min_table[row] -= min_of_row
                min_cost += min_of_row
        for column in range(min_table.shape[1]):  # n^2 checks at constant time.
            min_of_column = numpy.amin(min_table[:, column])
            if min_of_column != math.inf:
                min_table[:, column] -= min_of_column
                min_cost += min_of_column
        return min_cost

    # including the edge from city node_row to city node column, and then reduce the matrix.
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

    # Previous iterations of my program had each state hold onto a list of city object. Holding the indices sped
    # things up a bunch.
    def make_route(self, city_indices, all_cities):
        route = []
        for i in range(len(city_indices)):
            route.append(all_cities[city_indices[i]])
        return route

    ''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found during search, the
		best solution found.  You may use the other three field however you like.
		algorithm</returns>
	'''

    def fancy(self, time_allowance=60.0):  # i.e. 2opt
        startTime = time.time()
        solutions = []
        count = 0
        num_trials_without_update = 0
        # get the costs from each city to each other city and store as a matrix.
        cities = self._scenario.getCities()
        ncities = len(cities)
        best_of_all = math.inf
        greedy_solutions = []
        num_tests = int(max(200/ncities, 2))  # Haven't decided on the number of test we should do.
        tests_began = 0
        for tests in range(100):
            tests_began += 1
            bestSoFar = self.greedy(time_allowance)['soln']
            greedy_solutions.append(bestSoFar)
            if best_of_all > bestSoFar.cost:
                best_of_all = bestSoFar.cost
                count +=1

            keep_going = True
            while keep_going and time.time() - startTime < time_allowance:
                count_at_beginning = count
                keep_going = False
                # For loop that loops through the current route, and tries to swap things.
                for i in range(0, ncities - 1):  # Is this going to make a difference?
                    for j in range(i + 1, ncities):
                        new_route = self.two_opt_swap(bestSoFar.route, i, j)
                        new_solution = TSPSolution(new_route)
                        if new_solution.cost != math.inf:
                            if new_solution.cost < bestSoFar.cost:
                                # print(count)
                                if new_solution.cost < best_of_all:
                                    count = count + 1
                                    best_of_all = new_solution.cost
                                bestSoFar = new_solution
                                keep_going = True
                                break
                    if keep_going:
                        break
            print("Best of run: ", bestSoFar.cost)
            print("Very best run so far: ", best_of_all)
            solutions.append(bestSoFar)
            if len(solutions) >= 2:
                if count_at_beginning == count:
                    num_trials_without_update += 1
                if count_at_beginning != count:
                    num_trials_without_update = 0
            if num_trials_without_update >= 100:
                print("coulda ended ages ago.")
                break
                # if solutions[len(solutions)-1].cost == solutions[len(solutions) - 1]:
                #     break

            if time.time() - startTime > time_allowance:
                break
            print("Finished test #", tests)
            print("Count is ", count)
        get_sol = bestSoFar
        for solution in solutions:
            if get_sol.cost > solution.cost:
                get_sol = solution
        endTime = time.time()
        results = {}  # create a new results object.
        results['cost'] = get_sol.cost
        results['time'] = endTime - startTime
        results['count'] = count
        results['soln'] = get_sol
        results['max'] = None
        results['total'] = tests_began  # how many of our tests were actually completed in time.
        results['pruned'] = None
        return results

    def fastRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        num_cities_checked = 0
        city_starts = []

        while not foundTour and time.time() - start_time < time_allowance and len(city_starts) < ncities:
            for city in range(ncities):
                cities[city].visited = False
            route = []
            found_city = False
            while not found_city and len(city_starts) < ncities:
                num_cities_checked += 1
                rand_city = np.random.randint(0, ncities)
                if not city_starts.count(rand_city):
                    city_starts.append(rand_city)
                    curr_city_index = rand_city
                    found_city = True
            if len(city_starts) == ncities:
                break

            # curr_city_index = city_index
            # Add the starting city to the route, and show that we HAVE visited it.
            route.append(cities[curr_city_index])
            for num_visited in range(ncities):
                cities[curr_city_index].visited = True  # We will visit this city, set it to visited.
                for available_edges in range(ncities):
                    city_edges = []
                    i = None
                    found_edge = False
                    while not found_edge and len(city_edges) < ncities:
                        num_cities_checked += 1
                        i = np.random.randint(0, ncities)
                        if not city_edges.count(i):
                            city_edges.append(i)
                            curr_city_index = i
                            found_edge = True
                    new_dist = cities[curr_city_index].costTo(cities[i])
                    if not cities[i].visited and new_dist < np.inf:
                        curr_city_index = i
                        break

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

    def two_opt_swap(self, initial_route, start, end):
        swapped_route = []
        for i in range(0, start):
            swapped_route.append(initial_route[i])
        for i in range(end, start - 1, -1):
            swapped_route.append(initial_route[i])
        for i in range(end + 1, len(initial_route)):
            swapped_route.append(initial_route[i])
        return swapped_route



class State:
    def __init__(self, cost_table, lower_bound, depth, last):
        self.route = []
        self.cost_table = cost_table
        self.lower_bound = lower_bound
        self.depth = depth
        self.last = last

    # Overriding the less than function to give the state with the lowest priority value precedence.
    def __lt__(self, other):  # return true if other's priority is less than self's.
        priority_self = self.get_priority()
        priority_other = other.get_priority()
        if priority_other > priority_self:
            return True
        else:
            return False

    # Adjusting this function allows modifies which state it expands first, hopefully to calculate new solutions and
    # allow other states to be pruned sooner. I tried a number of different ways to determine priority,
    # and found that while depth should be considered, it should be limited so that if an expanded state has a cost
    # much worse than its parent, it won't have priority.
    def get_priority(self):  # Priority can be calculated at constant time.
        priority = self.lower_bound / (self.depth ** .8)
        return priority
