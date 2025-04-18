"""
directsearch
============

This is a Python package for solving unconstrained minimization problems, which only uses
function values (no derivatives needed).

It implements a family of algorithms of direct-search type. In addition to 
standard, deterministic frameworks, the package also contains randomized 
techniques based on generating directions in random subspaces.

===============================================================================
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

# Version identifier
__version__ = '1.0'
__all__ = ['__version__']

# Import the main routines
from .interface import solve, solve_directsearch, solve_probabilistic_directsearch, solve_subspace_directsearch, solve_stp
__all__ += ['solve', 'solve_directsearch', 'solve_probabilistic_directsearch', 'solve_subspace_directsearch', 'solve_stp']
