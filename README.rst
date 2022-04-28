==============================================================
directsearch: derivative-free optimization using direct search
==============================================================

.. image::  https://github.com/lindonroberts/directsearch/actions/workflows/unit_tests.yml/badge.svg
   :target: https://github.com/lindonroberts/directsearch/actions
   :alt: Build Status

.. image::  https://img.shields.io/badge/License-GPL%20v3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0
   :alt: GNU GPL v3 License

.. image:: https://img.shields.io/pypi/v/directsearch.svg
   :target: https://pypi.python.org/pypi/directsearch
   :alt: Latest PyPI version

``directsearch`` is a package for solving unconstrained minimization, without requiring derivatives of the objective. It is particularly useful when evaluations of the objective function are expensive and/or noisy.

It implements a family of direct search methods.
For general references on these methods, please consult:

1. A R Conn, K Scheinberg, and L N Vicente. *Introduction to derivative-free optimization*. SIAM, 2009.
2. C Audet, and W. Hare. Derivative-Free and Blackbox Optimization. Springer, 2017.
3. T G Kolda, R M Lewis, and V Torczon. Optimization by Direct Search: New Perspectives on Some Classical and Modern Methods. *SIAM Review*, 45(3), 2003, 385-482.

This package extends general direct search methods to use randomized methods for improved practical performance and scalability.

Citation
--------
If you use this package, please cite:

L Roberts, and C W Royer. Direct search based on probabilistic descent in reduced spaces, Technical report arXiv:2204.01275, April 2022.

Installation
------------
Please install using pip:

.. code-block:: bash

    $ pip install [--user] directsearch

To instead install from source run:

.. code-block:: bash

    $ git clone git@github.com:lindonroberts/directsearch.git
    $ cd directsearch
    $ pip install -e .

The ``-e`` option to pip allows you to modify the source code and for your Python installation to recognize this.

Usage
-----
This package can solve unconstrained nonlinear optimization problems of the form: ``min_{x in R^n} f(x)``.
The simplest usage of ``directsearch`` is

.. code-block:: python

    soln = directsearch.solve(f, x0)

where

* ``f`` is a callable objective function, taking in a ``numpy.ndarray`` the same shape as ``x0`` and returing a single ``float``.
* ``x0`` is a one-dimensional ``numpy.ndarray`` (i.e. ``len(x0.shape)==1``), the starting point for the algorithm. It should be the best available guess of the minimizer.

The output is an object with fields:

* ``soln.x``: the approximate minimizer, the best ``x`` value found (a ``numpy.ndarray`` the same shape as ``x0``).
* ``soln.f``: the minimum value equal to ``f(soln.x)``.
* ``soln.nf``: the number of evaluations of ``f`` required by the solve routine.
* ``soln.flag``: an integer indicating the reason for termination.
* ``soln.msg``: a string with a human-readable termination message.

The possible values of ``soln.flag`` are:

* ``soln.EXIT_MAXFUN_REACHED``: termination on maximum number of objective evaluations.
* ``soln.EXIT_ALPHA_MIN_REACHED``: termination on small step size (success).

You can print information about the solution using ``print(soln)``.
The ``examples`` directory has several scripts showing the usage of ``directsearch``.

**Interfaces to solver instances**

There are many configurable options for the solver in `directsearch` and several ways to call specific direct search algorithm implementations.
The full set of available functions is:

* ``directsearch.solve()`` applies a direct-search method to a given optimization problem. It is the most flexible available routine.
* ``directsearch.solve_directsearch()`` applies regular direct-search techniques without sketching [1,2,3].
* ``directsearch.solve_probabilistic_directsearch()`` applies direct search based on probabilistic descent without sketching [4].
* ``directsearch.solve_subspace_directsearch()`` applies direct-search schemes based on polling directions in random subspaces [5].
* ``directsearch.solve_stp()`` applies the stochastic three points method, a particular direct-search technique [6].

**Optional parameters and more information**

See ``usage.txt`` for full details on how to call these functions.
The most commonly used optional inputs (to all functions) are:

* ``maxevals``: the maximum number of allowed evaluations of ``f`` during the solve.
* ``verbose``: a ``bool`` for whether or not to print progress information.
* ``print_freq``: an ``int`` indicating how frequently to print progress information (1 is at every iteration).

**Choosing a solver instance**

As a rule of thumb, if ``len(x0)`` is not too large (e.g. less than 50), then ``solve_directsearch()`` or ``solve_probabilistic_directsearch()`` are suitable choices.
Of these, generally ``solve_probabilistic_directsearch()`` will solve with fewer evaluations of ``f``, but ``solve_directsearch()`` is a deterministic algorithm.
If ``len(x0)`` is larger, then ``directsearch.solve_subspace_directsearch()`` may be a better option.
Note that ``solve_directsearch()`` is the only deterministic algorithm (i.e. reproducible without setting the numpy random seed).

**References**

1. A R Conn, K Scheinberg, and L N Vicente. *Introduction to derivative-free optimization*. SIAM, 2009.
2. C Audet, and W. Hare. Derivative-Free and Blackbox Optimization. Springer, 2017.
3. T G Kolda, R M Lewis, and V Torczon. Optimization by Direct Search: New Perspectives on Some Classical and Modern Methods. *SIAM Review*, 45(3), 2003, 385-482.
4. S Gratton, C W Royer, L N Vicente, and Z Zhang. Direct Search Based on Probabilistic Descent. *SIAM J. Optimization*, 25(3), 2015, 1515-1541.
5. L Roberts, and C W Royer. Direct search based on probabilistic descent in reduced spaces, *In preparation*, (2022).
6. E H Bergou, E Gorbunov, and P Richtarik. Stochastic Three Points Method for Unconstrained Smooth Minimization. *SIAM J. Optimization*, 30(4), 2020, 2726-2749.

Bugs
----
Please report any bugs using GitHub's issue tracker.

License
-------
This algorithm is released under the GNU GPL license.
