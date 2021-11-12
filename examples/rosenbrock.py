#!/usr/bin/env python3

"""
Simple directsearch example: minimize the Rosenbrock function
"""

from __future__ import print_function
import numpy as np
import directsearch

# Define the objective function
def rosenbrock(x):
    return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2

# Define the starting point
x0 = np.array([-1.2, 1.0])

print("Using directsearch to minimize the Rosenbrock function")
print("Starting from f(x0) = %g" % rosenbrock(x0))
print("")

# Call solver
xmin, fmin, nf, msg = directsearch.solve(rosenbrock, x0)  # add verbose=True to print more information

# Display output
print("**** Basic direct search routine ****")
print("Found solution fmin = %g at" % fmin)
print("xmin =", xmin)
print("Took %g evaluations" % nf)
print("Output message: %s" % msg)

print("")

# Try again using a random direct search method
np.random.seed(0)  # for reproducibility
xmin2, fmin2, nf2, msg2 = directsearch.solve(rosenbrock, x0, poll_type='random2')

print("**** Probabilistic direct search routine ****")
print("Found solution fmin = %g at" % fmin2)
print("xmin =", xmin2)
print("Took %g evaluations" % nf2)
print("Output message: %s" % msg2)