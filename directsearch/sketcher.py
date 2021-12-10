"""
A collection of routines for generating random subspaces

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
from math import sqrt
import scipy.sparse as sparse

__all__ = ['sketch_matrix', 'check_valid_sketch_method']


def randint_without_replacement(q, m, s):
    '''
    Select s random values from [0,...,q-1] *without* replacement, done m times independently
    Return as s*m dense matrix of row indices in range(q)
    '''
    if s == 1:
        # Easy case
        return np.random.randint(0, q, size=m).reshape((1,m))
    elif s == 2:
        # Do the s=2 case cleverly without a slow Python 'for' loop over m
        subsample1 = np.random.randint(0, q, size=m)  # first row index for each column
        subsample2 = np.random.randint(0, q, size=m)  # second row index for each column
        # Since subsample1 and subsample2 mostly don't overlap (when q >> 2), we should just fix these entries
        overlaps_to_fix = np.where(subsample1 == subsample2)[0]
        new_subsample = np.random.randint(0, q-1, size=len(overlaps_to_fix))  # new values in range [0,...,q-2]
        new_subsample[new_subsample >= subsample1[overlaps_to_fix]] += 1  # change to range [0,...q-1] but != subsample1
        subsample2[overlaps_to_fix] = new_subsample
        return np.vstack([subsample1, subsample2])  # size s*m
    else:
        # Unfortunately, np.random.randint doesn't allow without-replacement sampling
        # The only way we can do this is with a Python loop over m (slow)
        indices = np.arange(q)
        return np.vstack([np.random.choice(indices, size=s, replace=False) for _ in range(m)]).T


def hashing_scalings(m, s):
    '''
    Return entries of q*m hashing matrix S (of which there are s*m values)
    Output is s*m dense matrix
    '''
    return (2 * np.random.randint(0, 2, size=m*s) - 1) / sqrt(s)  # random array of +/- 1/sqrt(s)


def sketch_hashing(q, m, s=2):
    '''
    A is (something * m) matrix, calculate A*S.T where S is a q*m hashing matrix (with given s)
    Build S in CSR format for fast S*x matrix-vector products (scipy recommends CSR over CSC for this)
    '''
    vals = hashing_scalings(m, s)
    rows = randint_without_replacement(q, m, s)
    cols = np.vstack([np.arange(m) for _ in range(s)])  # s * m matrix of column indices
    S = sparse.csr_matrix((vals.flatten(), (rows.flatten(), cols.flatten())), shape=(q, m))
    return S


def sketch_gaussian(q, m):
    '''
    A is (something * m) matrix, calculate A*S.T where S is a q*m Gaussian matrix
    '''
    S = np.random.normal(size=(q, m)) / sqrt(q)
    return S


def qr_positive_diagonal(A):
    '''
    QR factorization but where diag(R) > 0

    Based on the complex version (p11 of https://arxiv.org/pdf/math-ph/0609050.pdf)

    This was designed for A square, but also works for A tall & skinny
    '''
    Q, R = np.linalg.qr(A, mode='reduced')
    R_diag_signs = np.diag(R) / np.abs(np.diag(R))
    # TODO do scaling by R_diag_signs without doing full matrix-matrix multiplication
    Q2 = Q @ np.diag(R_diag_signs)
    R2 = np.diag(R_diag_signs) @ R
    return Q2, R2


def sketch_orthogonal(q, m):
    '''
    Simple orthogonal sketching from Haar measure

    Eq (10), https://arxiv.org/pdf/2003.02684.pdf
    '''
    Z = np.random.randn(m, q)  # save computation by dropping remaining columns of Z (get the same result in the end)
    Q, R = qr_positive_diagonal(Z)
    S = np.sqrt(m / q) * Q.T  # shape is q * m
    return S

def check_valid_sketch_method(sketch_method):
    if sketch_method.startswith('hashing'):  # e.g. hashing1
        try:
            s = int(sketch_method.replace('hashing', ''))
            return (s > 0)
        except ValueError:  # can't cast string for s to int
            return False
    elif sketch_method == 'gaussian':
        return True
    elif sketch_method == 'orthogonal':
        return True
    else:
        return False

def sketch_matrix(sketch_dimension, ambient_dimension, sketch_method):
    """
    Build a sketching matrix of size sketch_dimension*ambient_dimension
    """
    if sketch_method.startswith('hashing'):  # e.g. hashing1
        s = int(sketch_method.replace('hashing', ''))
        return sketch_hashing(sketch_dimension, ambient_dimension, s=s)
    elif sketch_method == 'gaussian':
        return sketch_gaussian(sketch_dimension, ambient_dimension)
    elif sketch_method == 'orthogonal':
        return sketch_orthogonal(sketch_dimension, ambient_dimension)
    else:
        raise RuntimeError("Unknown sketching method '%s'" % sketch_method)
