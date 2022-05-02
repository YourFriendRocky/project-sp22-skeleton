"""Solves an instance.

Modify this file to implement your own solvers.

For usage, run `python3 solve.py --help`.
"""
import argparse
from pathlib import Path
import sys
from typing import Callable, Dict
import numpy as np
from gekko import GEKKO
import gurobipy
import math
from point import Point
from instance import Instance
from solution import Solution
from file_wrappers import StdinFileWrapper, StdoutFileWrapper


def solve_naive(instance: Instance) -> Solution:
    return Solution(
        instance=instance,
        towers=instance.cities,
    )

def solve_GEKKO(instance: Instance) -> Solution:
    """Uses LP solver GEKKO to find a basic MILP solution to cities/generators problem
    @param: instance: the given input instance, has parameters grid_side_length, coverage_radius, penalty_radius, and cities
    @return: Solution
    """
    #given variables
    D = instance.grid_side_length
    Rp = instance.penalty_radius
    Rs = instance.coverage_radius
    cities = instance.cities
    m = GEKKO(remote=False)
    #DxD Variable Array (representative of graph)
    v = m.Array(m.Var, (instance.grid_side_length, instance.grid_side_length),lb=0,ub=1,integer=True)
    #Initial Guess (1 on each city, 0 everywhere else)
    for i in range(instance.grid_side_length):
        for j in range(instance.grid_side_length):
            v[i,j].value = 0
            v[i,j].upper = 1
            v[i,j].lower = 0

    for c in instance.cities:
        v[c.x,c.y].value = 1
        v[i,j].upper = 1
        v[i,j].lower = 0
    #define parameter
    one = m.Param(value = 1)
    zero = m.Param(value = 0)

    
    #Objective Function: multiple functions are added together
    #   Penalty equation: (ln(170) + .17(sum of all towers within penalty range)) * x(ij)  

    #goes through every single possible tower placement
    """
    for x in range(D):
        for y in range(D):
            w = []
            #goes through all points in the general area
            for i in range(max(0, x-Rp), min(Rp, x+Rp)):
                for j in range(max(0, y-Rp), min(Rp, y+Rp)):        
                    if (Point(i,j).distance_obj(Point(x,y)) <= Rp):
                        w.append(v[i,j])
            m.Minimize((170 * 2.7182 ** (.17 * m.sum(w))) * v[x,y])
    """
    #goes through every single tower placement within a city's radius
    included = {}
    for c in cities:
        x,y = c.x,c.y
        for i in range(max(0, x-instance.coverage_radius), min(instance.grid_side_length, x+instance.coverage_radius)):
            for j in range(max(0, y-instance.coverage_radius), min(instance.grid_side_length, y+instance.coverage_radius)):
                if(c.distance_obj(Point(i,j)) <= Rs):
                    if((i,j)) not in included:
                        included[(i,j)] = Point(i,j)
                        #goes through all points in the general area
                        w = []
                        for i in range(max(0, x-Rp), min(Rp, x+Rp)):
                            for j in range(max(0, y-Rp), min(Rp, y+Rp)):        
                                if (Point(i,j).distance_obj(Point(x,y)) <= Rp):
                                    w.append(v[i,j])
                        #m.Minimize((m.log(170) + (.17 * m.sum(w))) * v[x,y])
                        m.Minimize((170 * 2.7182 ** (.17 * m.sum(w))) * v[x,y])

    
    
    #Constraints; equation format: m.Equation(m.sum([xi**2 for xi in x])==40); m.Equation(np.prod(x) >= 25; x1 + x2 == 20)
    #at least one point within 3 of each city must be at least 1
    
    #instantiate np array of size(cities)
    #use gekko
    #equations = np.array([0 for i in range(len(instance.cities))])

    #Make sure every city is covered by at least one tower
    for c in instance.cities:
        x,y=c.x,c.y
        l= []
        for i in range(max(0, x-instance.coverage_radius), min(instance.grid_side_length, x+instance.coverage_radius)):
            for j in range(max(0, y-instance.coverage_radius), min(instance.grid_side_length, y+instance.coverage_radius)):
                #if i+j<=3 or (abs(i-x) == abs(j-y) and (abs(i-x) == 2 or abs(j-y) == 2)):
                #    l.append(v[i,j]) TODO: Fix this so that it's using dist instead
                if (c.distance_obj(Point(i, j)) <= instance.coverage_radius):
                    l.append(v[i,j])
        m.Equation(m.sum(l) >= one)

    #set solver to MINLP
    m.options.SOLVER = 1
    #Run solver
    m.solve()

    #solution list
    sol = []
    #return towers
    points_in_radius = 29
    for x in range(D):
        for y in range(D):
            if v[x,y].value[0] >= (1/points_in_radius):
                print(f"Tower exists at ({x},{y}): {v[x,y].value}")
                sol.append(Point(x,y))
    return Solution(
        instance = instance,
        towers = sol
    )
    
def solve_knapsack(instance: Instance) -> Solution:
    """Uses knapsack to find a minimizing solution in exponential time"""
    D = instance.grid_side_length
    cities = [c for c in instance.cities]
    towers = []
    for c in cities:
        x,y = c.x,c.y
        in_range = []
        for i in range(max(0, x-instance.coverage_radius), min(instance.grid_side_length, x+instance.coverage_radius)):
            for j in range(max(0, y-instance.coverage_radius), min(instance.grid_side_length, y+instance.coverage_radius)):
                if Point(i,j).distance_obj <= instance.coverage_radius:
                    in_range.append(Point(i,j))
        towers.append(in_range)
    

    return Solution(
        instance = instance,
        towers = solve_knapsack_helper(cities, towers, instance.penalty_radius, [])
    )


def solve_knapsack_helper(cities, towers_in_range, Rp, sol):
    """Tail recursive solution similar to the coin-choosing problem"""
    if cities is not True:
        return sol
    return min([solve_knapsack_helper( \
        cities[1:], towers_in_range[1:], sol.append(t)) for t in towers_in_range[0]], key = lambda g: knapsack_penalty(g, Rp))

    
def knapsack_penalty(towers, Rp):
        """Computes the penalty given penalty radius and number of towers."""
        penalty = 0
        for fidx, first in enumerate(towers):
            num_overlaps = 0
            for sidx, second in enumerate(towers):
                if fidx == sidx:
                    continue
                if Point.distance_obj(first, second) <= Rp:
                    num_overlaps += 1
            penalty += 170 * math.exp(0.17 * num_overlaps)
        return penalty

def solve_greedy(instance: Instance) -> Solution:
    #create a list of all cities; each time a city is fulfilled, remove it from the list
    cities = [c for c in instance.cities]
    towers = []
    #Set Variables from instance
    D = instance.grid_side_length
    Rp = instance.penalty_radius
    Rs = instance.coverage_radius

    #While loops that contineus until all cities are exausted from the cities list
    #(We covered every single city)
    while cities.len() != 0:
        #For every single while loop iteration we will pick a best tower to place
        #
        best_tower = None
        best_city_number = None
        curr_cities = []
        tower_tracker = {}
        for c in cities:
            x,y = c.x,c.y
            for i in range(max(0, x-instance.coverage_radius), min(instance.grid_side_length, x+instance.coverage_radius)):
                for j in range(max(0, y-instance.coverage_radius), min(instance.grid_side_length, y+instance.coverage_radius)):
                    if Point(i,j).distance_obj <= instance.coverage_radius:
                        if (i,j) not in tower_tracker:
                            tower_tracker[(i,j)] = [c]
                        else:
                            tower_tracker[(i,j)].append(c)
        best_tower = max(tower_tracker, key = len(tower_tracker.get))
        towers.append(Point(best_tower[0], best_tower[1]))
        for c in tower_tracker[best_tower]:
            cities.remove(c)
    return Solution(
        instance = instance,
        towers = towers
    )
            
    '''
        for x in range(D):
            for y in range(D):
                current_city_number = 0
                for city in cities:
                    if city.distance_obj(Point(x, y)) <= Rs:
                        curr_cities.append(cities.pop(city))
                        current_city_number += 1
                if (not best_tower):
                    best_tower = Point(x, y)
                    best_city_number = current_city_number
                elif (best_city_number < current_city_number):
                    best_tower = Point(x, y)
                    best_city_number = current_city_number
                else:
                    cities += curr_cities 
        towers.append(best_tower)
    '''
                 
                
                

                
    return Solution(
        instance = instance,
        towers = instance.cities
    )

def solve_gurobi(instance: Instance) -> Solution:
    return Solution(
        instance = instance,
        towers = instance.cities
    )

    D = instance.grid_side_length
    Rp = instance.penalty_radius
    Rs = instance.coverage_radius
    cities = instance.cities





    



SOLVERS: Dict[str, Callable[[Instance], Solution]] = {
    "naive": solve_naive,
    "GEKKO": solve_GEKKO
}


# You shouldn't need to modify anything below this line.
def infile(args):
    if args.input == "-":
        return StdinFileWrapper()

    return Path(args.input).open("r")

def outfile(args):
    if args.output == "-":
        return StdoutFileWrapper()

    return Path(args.output).open("w")

def main(args):
    with infile(args) as f:
        instance = Instance.parse(f.readlines())
        solver = SOLVERS[args.solver]
        solution = solver(instance)
        assert solution.valid()
        with outfile(args) as g:
            print("# Penalty: ", solution.penalty(), file=g)
            solution.serialize(g)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve a problem instance.")
    parser.add_argument("input", type=str, help="The input instance file to "
                        "read an instance from. Use - for stdin.")
    parser.add_argument("--solver", required=True, type=str,
                        help="The solver type.", choices=SOLVERS.keys())
    parser.add_argument("output", type=str, 
                        help="The output file. Use - for stdout.", 
                        default="-")
    main(parser.parse_args())
