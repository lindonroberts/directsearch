"""
DSPFD - Direct Search based on Probabilistic Feasible Descent

Implementation based on the original Matlab package, available here:
https://www.lamsade.dauphine.fr/%7Ecroyer/numerics.html

The algorithm is described in the paper
S. Gratton, C. W. Royer, L. N. Vicente, and Z. Zhang. Direct Search Based on Probabilistic Feasible Descent for Bound
and Linearly Constrained Problems. Computational Optimization and Applications, 72:3 (2019), pp. 525-559.
http://link.springer.com/10.1007/s10589-019-00062-4
"""
import numpy as np
from scipy.linalg import null_space, orth


def doubledescLI(B, A):
    """
    Computation of generators of a pointed cone by the double description  method revisited (Fukuda, Prodon - 1997).
    GB = doubledescLI(B,A) computes a minimal generating set for the cone in R^d defined as {x | B*x = 0, A*x >= 0}.

    Inputs:
    - B: m*d matrix of full row rank m < d representing linear equalities
    - A: n*d matrix of linear inequalities

    Outputs:
    - R: Generating set of the cone, matrix of size d*nb where nb is the number of generators
    """
    m, d = B.shape
    n = A.shape[0]
    assert A.shape[1] == d, "B and A must have the same second dimensions, B.shape = %s, A.shape = %s" % (str(B.shape), str(A.shape))

    Ed = np.eye(d)
    # Initial set
    if m > 0:
        ZB = null_space(B)  # size d*dim(null(B))
        kb = np.linalg.matrix_rank(B)  # TODO can this be computed from ZB.shape[1]
    else:
        kb = 0
        ZB = Ed

    R = np.hstack((ZB, -ZB))  # This is a generating set for null(B), size d*nb

    for j in range(n):
        v = A[j, :] @ R
        if j == 0:
            ka = kb
            J = []
        else:
            J = list(range(j))  # Matlab = 1:(j - 1)
            ka = kb + np.linalg.matrix_rank(A[J, :])

        Iplus = np.nonzero(v > 0)[0]
        Izero = np.nonzero(v == 0)[0]
        Iminus = np.setdiff1d(list(range(R.shape[1])), np.hstack((Iplus, Izero)))

        # Finding the adjacent vectors
        Adj = 0
        Rnew = np.zeros((d, 0), dtype=float)

        for ip in range(len(Iplus)):
            for im in range(len(Iminus)):
                r1 = R[:, Iplus[ip]]
                r2 = R[:, Iminus[im]]
                ar1 = A[:j, :] @ r1
                ar2 = A[:j, :] @ r2
                if len(J) > 0:
                    z1 = np.nonzero(ar1[J, :] == 0.0)
                    if len(z1) > 0:
                        k1 = np.linalg.matrix_rank(np.vstack((B, A[z1, :])))
                    else:
                        k1 = kb
                    z2 = np.nonzero(ar2[J, :] == 0.0)
                    if len(z1) > 0:
                        k2 = np.linalg.matrix_rank(np.vstack((B, A[z2, :])))
                    else:
                        k2 = kb
                    iz = np.intersect1d(z1, z2)
                    if len(iz) > 0:
                        kz = np.linalg.matrix_rank(np.vstack((B, A[iz, :])))
                    else:
                        kz = kb
                else:
                    k1 = kb
                    k2 = kb
                    kz = kb
                if k1 == ka - 1 and k2 == ka - 1 and kz == ka - 2:
                    # Adjacent directions
                    r3 = ar1[j] * r2 - ar2[j] * r1
                    if np.linalg.norm(r3) != 0.0:
                        Rnew = np.hstack((Rnew, r3))
                        Adj += 1

        # Building the new set
        R = np.hstack((R[:, Iplus], R[:, Izero], Rnew))

    return R


def calldoubledescLI(Ae, Ai, Ipos, Ineg, Z):
    """
    Computation of the generators of a cone obtained as the intersection
    of a subspace and (possibly) some half-spaces.
        Ys, Yc = calldoubledescLI(Ae, Ai, Ipos, Ineg, Z)
    returns a positive spanning set for the cone
        {x | Ae*x=0, [Ai]j*x>=0 for all j in Ipos, [Ai]_j*x<=0 for all j in Ineg}

    Inputs:
    - Ae: matrix of full row rank m < n defining linear equalities
    - Ai: matrix defining linear equalities
    - Ipos: index of components of the elements of the cone that must be non-negative
    - Ineg: index of components of the elements of the cone that must be non-positive
    - Z: orthonormal basis for the null space of Ae

    Outputs:
    - Ys: set of linear generators for the cone
    - Yc: set of positive generators for the cone
    """
    r = np.linalg.matrix_rank(Ae)
    if r < Ae.shape[0]:
        # Turning Ae into a full rank matrix
        Q = orth(Ae.T)
        Ae = Q.T

    W = Z @ Z.T @ Ai.T
    Aeplus = Ae
    Aiplus = np.hstack((-W[:, Ipos].T, W[:, Ineg].T))

    # Call the double description method to compute the generators of an appropriate cone
    G = doubledescLI(Aeplus, Aiplus)
    nb = G.shape[1]

    for i in range(nb):
        G[:, i] /= np.linalg.norm(G[:, i])

    # Identification of the lineality space
    Is = np.array([], dtype=int)
    Ir = np.array([], dtype=int)
    V = G.T @ G
    Iv = list(range(V.shape[0]))
    while len(Iv) > 0:
        i = Iv[0]
        V = G[:, i].T @ G
        Jopp = np.nonzero(V == -1.0)[0]
        if len(Jopp) > 0:
            Is = np.hstack((Is, [i]))
            Ir = np.hstack((Ir, Jopp))
            Iv = np.setdiff1d(Iv, Jopp)

        Jsame = np.nonzero(V == 1.0)[0]
        Jsame = np.setdiff1d(Jsame, [i])
        if len(Jsame) > 0:
            Iv = np.setdiff1d(Iv, Jsame)
            Ir = np.hstack((Ir, Jsame))

        Iv = np.setdiff1d(Iv, [i])

    Ic = np.setdiff1d(list(range(G.shape[1])), np.hstack((Is, Ir)))
    Ys = G[:, Is]
    Yc = G[:, Ic]
    return Ys, Yc


def main():
    # Testing here for now

    # Example from Dobler, 1994
    # C = {x in R^3 : x0 <= x1 <= x2 and x0=x2}
    d = 3
    B = np.array([[-1.0, 0.0, 1.0]])
    A = np.array([[1.0, -1.0, 0.0], [0.0, -1.0, 1.0]])
    print("B =", B)
    print("A =", A)
    Gb = doubledescLI(B, A)
    Gb_true = np.array([[0.7071, 0], [0, -1], [0.7071, 0]])  # from Matlab implementation
    print("Gb true =", Gb_true)
    print("Gb =", Gb)

    # TODO test calldoubledescLI
    # TODO unit tests?

    # TODO still to implement:
    # TODO - gentech.m  (compute generators of approx tangent cone)
    # TODO - activeconsAiZ.m (find almost active constraints)
    # TODO - randcolZsearch.m (random subset of generators of null(B), poll directions for equality only cons)
    print("Done")
    return


if __name__ == '__main__':
    main()
