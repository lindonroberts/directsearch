"""
A collection of routines for generating random subspaces.

- randint_without_replacement: Auxiliary routine to draw a subset of random 
indices

- hashing_scalings: Auxiliary routine used to generate hashing sketches.

- sketch_hashing: Computes a sketch matrix based on hashing.

- sketch_gaussian: Computes a Gaussian sketching matrix.

- qr_positive_diagonal: Auxiliary routine that computes a specific QR 
factorization.

- sketch_orthogonal: Computes a random orthogonal sketching matrix.

- check_valid_sketch_method: Checks whether a sketching technique is
implemented.

- sketch_matrix: 

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

# Useful imports
import numpy as np
from math import sqrt
import scipy.sparse as sparse

# Global variables
__all__ = ['sketch_matrix', 'check_valid_sketch_method']


###############################################################################
def randint_without_replacement(q, m, s):
    """
        Selecting a random subset of indices.

        randint_without_replacement(q,m,s) constructs a matrix whose columns 
        are indices drawn without replacement in {0,...,q-1}.

        Inputs:
            q: Range for the indices.
            m: Number of independent trials.
            s: Number of values drawn during each trail
        
        Output: An s*m dense matrix of row indices in range(q)
    """
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
        # NOTE: Could we try to reason on a matrix of indices
        # np.random.rand(m,q).argsort(1)[:,:s] ?
        indices = np.arange(q)
        return np.vstack([np.random.choice(indices, size=s, replace=False) for _ in range(m)]).T

###############################################################################
def hashing_scalings(m, s):
    """
        Generation of scaling matrix for hashing purposes.

        hashing_scaling(m,s) constructs an (s,m) matrix with +/- 1/sqrt(s) 
        coefficients.

        Inputs:
            m: Number of rows of the matrix.
            s: Number of values drawn during each trail
        
        Output: An s*m dense matrix of row indices in range(q)
    """
    return (2 * np.random.randint(0, 2, size=m*s) - 1) / sqrt(s)  

###############################################################################
def sketch_hashing(q, m, s=2):
    """
        Builds a hashing sketching matrix S to define a column sketch operator
        A -> A*S.T

        sketch_hashing(q,m,s) produces a (q,m) matrix based on an (s,m) 
        matrix with +/-1/sqrt(s) elements.

        Inputs:
            q: Number of rows of the sketch
            m: Number of columns of the sketch
            s: Used to define the underlying hashing matrix.
                Default: 2

        Output:
            S: Sketching q-by-m matrix build in CSR format for fast S*x 
            matrix-vector products.
    """
    vals = hashing_scalings(m, s)
    rows = randint_without_replacement(q, m, s)
    cols = np.vstack([np.arange(m) for _ in range(s)])  # s * m matrix of column indices
    # scipy recommends CSR over CSC for fast S*x matvec
    S = sparse.csr_matrix((vals.flatten(), (rows.flatten(), cols.flatten())), shape=(q, m))
    return S

###############################################################################
def sketch_gaussian(q, m):
    """
        Builds a Gaussian sketching matrix S to define a column sketch
        A -> A*S.T

        sketch_gaussian(q,m) generates a matrix of size (q,m) with normally 
        distributed entries with zero mean and (1/sqrt(q)) variance.

        Inputs:
            q: Number of rows of the sketching matrix.
            m: Number of columns of the sketching matrix.

        Output: 
            S: A q-by-m matrix with Gaussian entries.
    """
    S = np.random.normal(size=(q, m)) / sqrt(q)
    return S

###############################################################################
def qr_positive_diagonal(A):
    '''
        Adjusted QR factorization.

        Q2,R2 = qr_positive_diagonal(A) outputs a QR factorization Q2*R2 = A 
        such that all coefficients on the diagonal of R2 are positive.

        Inputs:
            A: Matrix to be factorized, either square or with more rows than 
            columns
        
        Outputs:
            Q2: Orthogonal factor in the QR factorization
            R2: Triangular factor in the QR factorization with positive 
            diagonal elements.

        Source:
             https://arxiv.org/pdf/math-ph/0609050.pdf, page 11.
    '''
    Q, R = np.linalg.qr(A, mode='reduced')
    R_diag_signs = np.diag(R) / np.abs(np.diag(R))
    # TODO do scaling by R_diag_signs without doing full matrix-matrix multiplication
    Q2 = np.dot(Q, np.diag(R_diag_signs))
    R2 = np.dot(np.diag(R_diag_signs), R)
    return Q2, R2

###############################################################################
def sketch_orthogonal(q, m):
    """
        Builds an orthogonal sketching matrix drawn from the Haar distribution.

        Inputs:
            q: Number of rows of the sketch.
            m: Number of columns of the sketch.

        Output: 
            S: q-by-m sketching matrix.

        Source: Eq (10), https://arxiv.org/pdf/2003.02684.pdf
    """
    Z = np.random.randn(m, q)  # save computation by dropping remaining columns of Z (get the same result in the end)
    Q, R = qr_positive_diagonal(Z)
    S = np.sqrt(m / q) * Q.T  # shape is q * m
    return S

###############################################################################
#NOTE: Is this method actually used?
def check_valid_sketch_method(sketch_method):
    """
       Checks whether a proposed sketching technique is implemented.

        Input:
            sketch_method: String representing a sketching technique

        Output: Boolean (True if the sketching technique is valid).
    """
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

###############################################################################
def sketch_matrix(sketch_dimension, ambient_dimension, sketch_method):
    """
        A wrapper for building a sketching matrix.

        sketch_matrix(sketch_dimension,ambient_dimension,sketch_method) builds
        a sketching matrix of size sketch_dimension*ambient_dimension. The 
        nature of this sketch depends on the parameter sketch_method.

        Inputs:
            sketch_dimension: Number of rows of the sketch
            ambient_dimension: Number of columns of the sketch
            sketch_method: Nature of the sketching atrix

        Output: A sketching matrix of size (sketch_dimension,ambient_dimension).
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
