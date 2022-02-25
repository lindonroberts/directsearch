#!/usr/bin/env python3

"""
Use directsearch to solve a nonconvex, robust linear regression problem.

Problem Source: Section 5 of
Carmon, Hinder, Duchi & Sidford (2017). "Convex Until Proven Guilty": Dimension-Free Acceleration of Gradient Descent on
Non-Convex Functions, https://arxiv.org/abs/1705.02766
"""
from __future__ import print_function
import numpy as np
import directsearch


def generate_data(d, m, seed=0):
    '''
    Generate a dataset amenable to nonconvex linear regression.

    Inputs:
        d: Dimension of the parameter space
        m: Number of data points
        seed: Seed for random number generator

    Outputs:
        A: m x d matrix
        b: Real vector with m components
    '''
    np.random.seed(seed)
    A = np.random.multivariate_normal(np.zeros(d), np.identity(d), size=m)
    z = np.random.multivariate_normal(np.zeros(d), 4 * np.identity(d))
    nu1 = np.random.multivariate_normal(np.zeros(m), np.identity(m))
    nu2 = np.random.binomial(1, 0.3, m)
    b = A.dot(z) + 3 * nu1 + nu2
    return A, b


def biweight_loss(t):
    """
    Computation of the biweight loss function.

    Inputs:
        t: float or array of real numbers

    Output:
        Float or array with loss function at each value of t
    """
    return t**2 / (1.0 + t**2)


# Create the dataset and objective function
d = 100  # dimension of dataset
m = 200  # number of data points
seed = 0  # for reproducibility
A, b = generate_data(d, m, seed=seed)
objfun = lambda x: np.mean(biweight_loss(A.dot(x) - b))

# Define the starting point
x0 = np.zeros((d,))

# Run solver
# This is a large-scale problem (len(x0) is large), so use the subspace version of directsearch
print("Using directsearch to minimize a nonconvex regression problem")
print("Starting from f(x0) = %g" % objfun(x0))
print("")

soln = directsearch.solve_subspace_directsearch(objfun, x0, sketch_dim=1, verbose=True)

# Display output
print(soln)

# Plot a summary of the results
import matplotlib.pyplot as plt
plt.plot((A@soln.x - b)/b, label='Relative error in residual')
plt.xlim(0, m)
plt.xlabel('Data element')
plt.ylabel('Relative residual error')
plt.title('Nonconvex regression: residual errors')
plt.grid()
plt.show()
