"""
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

# Ensure compatibility with Python 2
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import unittest

import directsearch


def array_compare(x, y, thresh=1e-14):
    return np.max(np.abs(x - y)) < thresh


def rosenbrock(x):
    return 10.0 * (x[1]-x[0]**2)**2 + (1-x[0])**2


class TestSolveRosenbrock(unittest.TestCase):
    # Minimise the (2d) Rosenbrock function
    def runTest(self):
        # n, m = 2, 2
        x0 = np.array([-1.2, 1.0])
        np.random.seed(0)
        for solve_fn in [directsearch.solve, directsearch.solve_directsearch, directsearch.solve_probabilistic_directsearch, directsearch.solve_subspace_directsearch, directsearch.solve_stp]:
            solver_name = solve_fn.__name__
            soln = solve_fn(rosenbrock, x0, maxevals=4000)
            print("%s results..." % solver_name)
            print(soln)
            self.assertTrue(array_compare(soln.x, np.array([1.0, 1.0]), thresh=1e-3), "Wrong xmin (%s)" % (solver_name))
            self.assertTrue(array_compare(soln.f, rosenbrock(soln.x), thresh=1e-10), "Wrong fmin compared to xmin (%s)" % (solver_name))
            self.assertTrue(abs(soln.f) < 1e-6, "Wrong fmin (%s)" % (solver_name))


class TestSolveLinear(unittest.TestCase):
    # Solve min_x ||Ax-b||^2, for some random A and b
    def runTest(self):
        n, m = 2, 5
        np.random.seed(0)  # (fixing random seed)
        A = np.random.rand(m, n)
        b = np.random.rand(m)
        sumsq = lambda v: np.dot(v, v)
        objfun = lambda x: sumsq(np.dot(A, x) - b)
        xmin = np.linalg.lstsq(A, b, rcond=None)[0]
        fmin = objfun(xmin)
        x0 = np.zeros((n,))
        np.random.seed(0)
        for solve_fn in [directsearch.solve, directsearch.solve_directsearch, directsearch.solve_probabilistic_directsearch, directsearch.solve_subspace_directsearch, directsearch.solve_stp]:
            solver_name = solve_fn.__name__
            if 'stp' in solver_name:
                # Need lower thresholds for STP
                soln = solve_fn(objfun, x0, maxevals=3000)
                print("%s results..." % solver_name)
                print(soln)
                self.assertTrue(array_compare(soln.x, xmin, thresh=1e-3), "Wrong xmin (%s)" % (solver_name))
                self.assertTrue(abs(soln.f - fmin) < 1e-3, "Wrong fmin (%s)" % (solver_name))
            else:
                soln = solve_fn(objfun, x0)
                print("%s results..." % solver_name)
                print(soln)
                self.assertTrue(array_compare(soln.x, xmin, thresh=1e-4), "Wrong xmin (%s)" % (solver_name))
                self.assertTrue(abs(soln.f - fmin) < 1e-4, "Wrong fmin (%s)" % (solver_name))