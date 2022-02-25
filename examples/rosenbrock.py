#!/usr/bin/env python3

"""
Simple directsearch example: minimize the Rosenbrock function
"""

from __future__ import print_function
import numpy as np
import directsearch

# Define the objective function
def rosenbrock(x):
    return 10.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2

# Define the starting point
x0 = np.array([-1.2, 1.0])

print("Using directsearch to minimize the Rosenbrock function")
print("Starting from f(x0) = %g" % rosenbrock(x0))
print("")

# Call solver
soln = directsearch.solve_directsearch(rosenbrock, x0, verbose=True)  # add verbose=True to print more information

# Display output
print(soln)

print("")

# Try again using a random direct search method
np.random.seed(0)  # for reproducibility
soln2 = directsearch.solve_probabilistic_directsearch(rosenbrock, x0)

print(soln2)
