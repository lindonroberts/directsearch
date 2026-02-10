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
    return fx

# Define the starting point and constraints (0 = bounds, 1 = linear inequality)
# Constraints are given as a matrix A and vector b, such that A @ x <= b
# If the initial point x0 does not satisfy the constraints, it is projected
# into the feasible region using scipy.optimize.minimize
constraint_choice = 1
if constraint_choice == 0:
    print("Bound constraints: x[0] <= 0.5, x[1] <= 0.7 --> minimizer is near [0.5, 0.25]")
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([0.5, 0.7])
elif constraint_choice == 1:
    print("Linear constraint: x[0] + x[1] <= 1 --> minimizer is near [0.62545, 0.37455]")
    A = np.array([[1.0, 1.0]])
    b = np.array([1.0])
else:
    raise RuntimeError("Unknown constraint_choice = %g" % constraint_choice)
print("")

# If the initial point x0 does not satisfy the constraints, it is projected
# into the feasible region using scipy.optimize.minimize
x0 = np.array([-1.2, -1.0])  # feasible initial point
# x0 = np.array([1.0, 1.0])  # infeasible initial point

# Call solver
soln = directsearch.solve_directsearch(rosenbrock, x0, A, b)  # add verbose=True to print more information

# Display output
print(soln)
