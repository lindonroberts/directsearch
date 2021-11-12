==============================================================
directsearch: derivative-free optimization using direct search
==============================================================

.. image::  https://img.shields.io/badge/License-GPL%20v3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0
   :alt: GNU GPL v3 License

directsearch is a package for solving unconstrained minimization, without requiring derivatives of the objective. It is particularly useful when evaluations of the objective function are expensive and/or noisy.

It implements a family of direct search methods.
For general references on these methods, please consult:

1. A R Conn, K Scheinberg, and L N Vicente. *Introduction to derivative-free optimization*. SIAM, 2009.
2. C Audet, and W. Hare. Derivative-Free and Blackbox Optimization. Springer, 2017.
3. T G Kolda, R M Lewis, and V Torczon. Optimization by Direct Search: New Perspectives on Some Classical and Modern Methods. *SIAM Review*, 45(3), 2003, 385-482.

This package extends general direct search methods to use randomized methods for improved practical performance and scalability.

Citation
--------
If you use this package, please cite:

L Roberts, and C W Royer. Direct search based on probabilistic descent in reduced spaces, *In preparation*, (2021).

Installation
------------
Please install using pip:

.. code-block:: bash

    $ pip install [--user] directsearch

Usage
-----
TODO

Bugs
----
Please report any bugs using GitHub's issue tracker.

License
-------
This algorithm is released under the GNU GPL license.
