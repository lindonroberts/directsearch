#!/usr/bin/env python3

"""
Simple directsearch example: minimize the Rosenbrock function with linear constraints
"""

from __future__ import print_function
import numpy as np
import directsearch

# Define the objective function
def rosenbrock(x):
    fx = 10.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2
    # print("x =", x, ", fx = %g" % fx)
    return fx

# Define the starting point and constraints (0 = bounds, 1 = linear inequality)
constraint_choice = 1
if constraint_choice == 0:
    # x[0] <= 0.5, x[1] <= 0.7 --> minimizer is approx (0.50, 0.25)
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([0.5, 0.7])
    x0 = np.array([-1.2, -1.0])  # starting point must be feasible
elif constraint_choice == 1:
    # x[0] + x[1] <= 1 --> minimizer is approx (0.62545, 0.37455)
    A = np.array([[1.0, 1.0]])
    b = np.array([1.0])
    x0 = np.array([-1.2, -1.0])  # starting point must be feasible
else:
    raise RuntimeError("Unknown constraint_choice = %g" % constraint_choice)

print("Using directsearch to minimize the Rosenbrock function with constraints")
print("Starting from f(x0) = %g" % rosenbrock(x0))
print("")

# Call solver
return_iteration_counts = True
if return_iteration_counts:
    soln, iter_counts = directsearch.solve_directsearch(rosenbrock, x0, A, b, verbose=True, return_iteration_counts=True)  # add verbose=True to print more information
else:
    soln = directsearch.solve_directsearch(rosenbrock, x0, A, b, verbose=True)  # add verbose=True to print more information
    iter_counts = None

# Display output
print(soln)
if iter_counts is not None:
    print(iter_counts)
