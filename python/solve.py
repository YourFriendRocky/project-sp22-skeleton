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

from instance import Instance
from solution import Solution
from file_wrappers import StdinFileWrapper, StdoutFileWrapper


def solve_naive(instance: Instance) -> Solution:
    return Solution(
        instance=instance,
        towers=instance.cities,
    )
"""Uses LP solver GEKKO to find a basic MILP solution to cities/generators problem

@param: instance: the given input instance, has parameters grid_side_length, coverage_radius, penalty_radius, and cities
@return: Solution
"""
def solve_GEKKO(instance: Instance) -> Solution:
    m = GEKKO()
    #DxD Variable Array (representative of graph)
    v = m.Array(m.Var, (instance.grid_side_length, instance.grid_side_length),lb=0,ub=1,integer=True)
    #Initial Guess (1 on each city, 0 everywhere else)
    for i in range(instance.grid_side_length):
        for j in range(instance.grid_side_length):
            v[i,j].value = 0

    for c in instance.cities:
        v[c.x,c.y].value = 1

    #define parameter
    one = m.Param(value=1)
    
    #Objective Function: multiple functions are added together
    #   Penalty equation: (ln(170) + .17(sum of all towers within penalty range)) * x(ij)
    

    
    
    #Constraints; equation format: m.Equation(m.sum([xi**2 for xi in x])==40); m.Equation(np.prod(x) >= 25; x1 + x2 == 20)
    #at least one point within 3 of each city must be at least 1

    #instantiate np array of size(cities)
    equations = np.array([0 for i in range(len(instance.cities))])

    biglist=[]
    for c in instance.cities:
        x,y=c.x,c.y
        l=[]
        for i in range(max(0, x-3), min(instance.grid_side_length, x+3)):
            for j in range(max(0, y-3), min(instance.grid_side_length, y+3)):
                #if i+j<=3 or (abs(i-x) == abs(j-y) and (abs(i-x) == 2 or abs(j-y) == 2)):
                #    l.append(v[i,j]) TODO: Fix this so that it's using dist instead
                i=1

        biglist.append(l)

    for i in range(len(biglist)):
        equations[i] = m.Equation(np.sum(biglist[i]) >= one)   

    #uses gekko Equations() function to instantiate the np array into satisfactory Equation objects
    equations = m.Equations(equations)




    



SOLVERS: Dict[str, Callable[[Instance], Solution]] = {
    "naive": solve_naive
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
