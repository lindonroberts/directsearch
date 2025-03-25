"""
DSPFD - Direct Search based on Probabilistic Feasible Descent

Implementation based on the original Matlab package, available here:
https://www.lamsade.dauphine.fr/%7Ecroyer/numerics.html

The algorithm is described in the paper
S. Gratton, C. W. Royer, L. N. Vicente, and Z. Zhang. Direct Search Based on Probabilistic Feasible Descent for Bound
and Linearly Constrained Problems. Computational Optimization and Applications, 72:3 (2019), pp. 525-559.
http://link.springer.com/10.1007/s10589-019-00062-4
"""
from math import ceil, log
import numpy as np
from scipy.linalg import null_space, orth


def doubledescLI(B, A):
    """
    Computation of generators of a pointed cone by the double description  method revisited (Fukuda, Prodon - 1997).
    GB = doubledescLI(B,A) computes a minimal generating set for the cone in R^d defined as {x | B*x = 0, A*x >= 0}.

    Reference:
    K. Fukuda and A. Prodon. Double description method revisited.
    In: Combinatorics and Computer Science (M. Deza, R. Euler, I. Manoussakis eds), Springer 1996

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


def gentech(Ae, Ai, Z, Ap, Am):
    """
    Computation of the generators of the approximate tangent cone as
    defined in Griffin et al (SISC, 2008).

    Ref:
    J. D. Griffin, T. G. Kolda and R. M. Lewis. Asynchronous Parallel Generating Set Search for Linearly Constrained
    Optimization. SIAM J. Scientific Computing, 30:4 (2008), pp. 1892-1924.

    Ys, Yc, calldd = gentech(Ae, Ai, Z, Ap, Am)
    returns the generators of the polar of a cone spanned by a linearly independent set of the columns of
    [Ae.T Ai[Ap,:] -Ai[Am,:]] that violate the constraints at (x,alpha).

    Inputs:
    - Ae: equality constraint matrix
    - Ai: inequality constraint matrix
    - Z: orthonormal basis for the null space of A
    - Ap: upper inequalities violated for the current iterate/step size
    - Am: lower inequalities violated for the current iterate/step size

    Outputs:
    - Ys: Linear generators for the polar of a cone defined by a subset of the columns of [A.T I -I]
    - Yc: Positive generators for the same polar
    - calldd: boolean indicating a call to the double description method
    """
    n = Z.shape[0]
    En = np.eye(n)
    calldd = False

    r = np.linalg.matrix_rank(Ae)
    if r < Ae.shape[0]:
        # Modifying Ae to obtain a full rank matrix
        Q = orth(Ae.T)
        Ae = Q.T

    Ib = np.intersect1d(Ap, Am)
    Iu = np.setdiff1d(Ap, Ib)
    Il = np.setdiff1d(Am, Ib)

    W = Z @ Z.T @ Ai.T
    Vp = np.array((n, 0), dtype=float)
    if len(Iu) > 0:
        Vp = np.hstack((Vp, W[:, Iu]))

    if len(Il) > 0:
        Vp = np.hstack((Vp, -W[:, Il]))

    Vl = Ae.T.copy()
    if len(Ib) > 0:
        Vl = np.hstack((Vl, W[:, Ib]))
        Zl = null_space(Vl.T)
    else:
        Zl = Z

    # Determine if a call to the double description is necessary
    if len(Vp) == 0:
        Ys = Zl
        Yc = np.array((n, 0), dtype=float)
    elif len(Zl) == 0:
        Ys = np.array((n, 0), dtype=float)
        Yc = np.array((n, 0), dtype=float)
    else:
        Q = Zl.T @ Vp
        rp = np.linalg.matrix_rank(Q)
        sp = Q.shape
        if rp == min(sp):
            # Non-degenerate case - Direct computation of the generators
            Qi, Ri = np.linalg.qr(Q)
            if sp[0] > sp[1]:
                Ys = Zl @ Qi[:, sp[1]+1:]
            else:
                Ys = np.array((n, 0), dtype=float)
            r2 = min(sp[1], Qi.shape[1])
            Qi = Qi[:, :r2]
            Ri = Ri[:r2, :r2]
            Yc = -Zl @ np.linalg.solve(Qi.T, Ri).T  # Yc = -Zl * (Qi / Ri.T) in original Matlab
            # Note: X = A/B in Matlab means solve X @ B = A
            # https://au.mathworks.com/matlabcentral/answers/2292-matrix-division-how-does-it-work
        else:
            # Degenerate case - calling the double description method
            Ys, Yc = calldoubledescLI(Ae, Ai, Ap, Am, Z)
            calldd = True

    return Ys, Yc, calldd


def activeconsAiZ(x, alpha, li, ui, Ai, Z, tolfeas=1e-15):
    """
    Computation of the index of active linear constraints at a given point.

    [Iap,Iam,nbgen] = activeboundsZ(x,alpha,li,ui,Ai,Z) returns the
    indexes of columns of [Ai -Ai] for which a displacement from a step
    length ALPHA is sufficiently feasible.

    Inputs:
    - x: current point
    - alpha: current step length
    - li: vector of lower bounds (LI(i) = -1e20 if unbounded)
    - ui: vector of upper bounds (UI(i) = +1e20 if unbounded)
    - Ai: matrix of linear inequality constraints and/or bounds
    - Z: orthonormal basis for a subspace of R^n, with n=dim(X)
    - tolfeas: feasibility tolerance for defining the approximately active constraints (initial choice of MATLAB patternsearch was 1e-3)

    Outputs:
    - Iap: indexes of columns of AI that cannot be used
    - Iam: Indexes of columns of -AI that cannot be used
    - nbgen: Number of generators
    """
    # n = len(x)
    # W = Z @ Z.T @ Ai.T
    nz = np.sqrt(np.diag(Ai @ Z @ Z.T @ Ai.T))
    iz = np.nonzero(nz < 1e-15)[0]

    ni = Ai.shape[0]
    # Eni = np.eye(ni)

    Vp = np.zeros((ni,), dtype=float)
    Vm = np.zeros((ni,), dtype=float)
    for i in range(ni):
        Vp[i] = -(np.dot(Ai[i, :], x) - ui[i] - tolfeas) / nz[i]
        Vm[i] = (np.dot(Ai[i, :], x) - li[i] + tolfeas) / nz[i]

    if len(iz) > 0:
        for i in range(len(iz)):
            j = iz[i]
            if abs(np.dot(Ai[j, :], x) - ui[j]) < tolfeas:
                Vp[j] = 0.0
            else:
                Vp[j] = np.inf

            if abs(np.dot(Ai[j, :], x) - li[j]) < tolfeas:
                Vm[j] = 0.0
            else:
                Vm[j] = np.inf

    alpha = min(alpha, 1e-3)
    Iap = np.nonzero(Vp <= alpha)[0]
    Iam = np.nonzero(Vm <= alpha)[0]
    nbgen = len(Iap) + len(Iam)
    return Iap, Iam, nbgen


def randcolZsearch(Z, Ip, Im, nbgen):
    """
    Random choice of columns of a given matrix Z to obtain a set of directions.

    D = randcolZsearch(Z,Ip,Im,nbgen) selects randomly nbgen elements among a subset of vectors in Z and -Z.

    Inputs:
    - Z: Matrix/Basis of the vectors to be used
    - Ip: Indexes of the inactive columns of the matrix Z
    - Im: Indexes of the inactive columns of the opposite matrix -Z
    - nbgen: Number of directions to be generated

    Output:
    - D: The set of generated directions
    """
    Y = np.hstack((Z[:, Ip], -Z[:, Im]))
    return Y[:, np.random.choice(Y.shape[1], nbgen, replace=False)]


def dspfdalgo(fun, x0, alpha0, dirgen, lb, ub, Aeq, beq, Aineq, lbineq, ubineq, seed, maxevals, ftarget, tolf, tol_feas):
    """
    Direct Search based on Probabilistic Feasible Descent.

    Implementation of direct-search methods based upon active-set
    considerations, with the aim of solving minimization problems with
    variables subject to bound and linear equality constraints.

    xsol, fsol, exitflag, output = dspfd(fun,x0,alpha0,dirgen,lb,ub,Aeq,beq,Aineq,lbineq,ubineq,seed,maxevals,ftarget,tolf,tol_feas)
    attempts to solve the optimization problem
    	minimize f(x) subject to 	Aeq*x=beq
        							lbi <= Aineq*x <= ubi
    								lb <= x <= ub.

    Inputs:
    - fun: The objective function
    - x0: Initial (feasible) point
    - alpha0: Initial step size
    - dirgen: Method that is used for generating the directions with
        0: Deterministic variant - always choose the generators of an approximate tangent cone (but randomly ordered)
        1: Proceed with a random subset selected within the generators of an approximate tangent cone
        2: Attempt to benefit from linear subspaces included in an approximate tangent cone to reduce the size of
           the (random) polling set, otherwise draw directions similarly to 1
    - lb, ub: Lower and upper bound vectors. Some components might be unbounded, in which case the corresponding bound
              is equal to +/- 1e20 or +/- Inf
    - Aeq: matrix (should be of full row rank) associated to the m<n equality constraints
    - beq: vector of size m associated to the m<n equality constraints
    - Aineq: matrix associated to linear inequality constraints
    - lbineq, ubineq: lower and upper bound vectors corresponding to the linear inequality constraints
    - seed: Integer used to initialize the random number generator
    - maxevals: Maximum number of function evaluations allowed
    - ftarget: Target value for the method
    - tolf: Tolerance with respect to the target value
    - tol_feas: Feasibility tolerance to consider a point as feasible

    Outputs:
    - xsol, fsol : Best point xsol and its value fsol found by the method
    - exitflag: flag indicating what happened in the method
    	0: Method stopped because the target function value was reached
        1: Method stopped because the step size was too small (expected behaviour)
        2: Method stopped because the budget of function evaluations was exceeded
        3: No feasible point was found past the initial one
        -1: An error occurred and the method did not complete its task
    - output: structure containing the following information
        funcCount: number of function evaluations performed in total
        numIter: number of iterations
        meanDirgen: number of directions generated per iteration
        meanFeaspts: mean number of feasible points per iteration
        fracDD: fraction of iterations calling the description method
        histF: history matrix of size (numIter+1)*2
            histF[:,0]: function values at the current iterate
            histF[:,1]: number of function evaluations
    """
    # Initialization and setting of auxiliary parameters
    n = len(x0)
    np.random.seed(seed)
    In = np.eye(n)
    exitflag = -1
    tol_alpha = alpha0 * (1e-6)
    #tol_feas = 1e-8
    #tol_feas = 1e-3

    # Step size update parameters and corresponding probability of descent
    if dirgen > 0:
        gamma = 2.0
    else:
        gamma = 1.0
        # Other possible value
        # gamma = 2.0
    theta = 0.5
    p0 = log(theta) / log(theta / gamma)
    # Minimum number of directions required in the unconstrained case
    nbunc = ceil(log(1.0 - log(theta) / log(gamma)) / log(2))

    # Counters and indexes
    funcCount = 0
    numIter	= 0
    numFeaspts = 0
    numDirgen = 0
    sumdd = 0

    # Before the loop
    x = x0
    f = fun(x)
    diff0 = tolf * (f-ftarget)
    funcCount = funcCount + 1
    histF = np.array([f, funcCount])
    alpha = alpha0
    alphamax = 20.0 * alpha0
    stopcrit = False

    # Determination of the presence of bounds
    lowbnd = np.nonzero(lb > -1e20)[0]
    upbnd = np.nonzero(ub < 1e20)[0]
    presbounds = len(upbnd) > 0 or len(lowbnd) > 0

    # Checking for equality constraints
    if len(Aeq) > 0:
        presleq = True
        for i in range(Aeq.shape[0]):
            auxa = np.linalg.norm(Aeq[i, :])
            Aeq[i, :] /= auxa
            beq[i] /= auxa
        Z = null_space(Aeq)
        m = Z.shape[1]
        if m == 0:
            # We are already at the solution, or the problem is infeasible
            stopcrit = True
    else:
        presleq = False
        Z = In
        m = n

    # Gathering linear inequalities and bounds
    if len(Aineq) > 0:
        preslineq = True
        for i in range(Aineq.shape[0]):
            auxa = np.linalg.norm(Aineq[i,:])
            Aineq[i, :] /= auxa
            lbineq[i] /= auxa
            ubineq[i] /= auxa
        Ai = np.vstack((Aineq, In))
        li = np.hstack((lbineq, lb))
        ui = np.hstack((ubineq, ub))
        ni = len(ui)
    else:
        preslineq = False
        Ai = In
        li = lb
        ui = ub
        ni = n

    # The projected normal vectors
    W = Z @ Z.T @ Ai.T

    # Main loop
    while not stopcrit:
        numIter += 1

        # Computation of the forcing function at the current iterate
        rho = 1e-4*(alpha**2)
        # rho = min(1e-4, 1e-4*(alpha**2))

        # Computation of the alpha-active and inactive constraints
        Iap, Iam, nbgen = activeconsAiZ(x, alpha, li, ui, Ai, Z)
        Iip = np.setdiff1d(list(range(ni)), Iap)
        Iim = np.setdiff1d(list(range(ni)), Iam)
        Ya = np.array((n,0), dtype=float)

        # Computation of the direction sets
        if nbgen == 0:
            # No active bounds - situation similar to linear equality-constrained (or unconstrained) case

            if dirgen == 0:
                # Positive spanning set corresponding to the null space of the linear equality constraints (if any)
                if not preslineq and not presbounds:
                    D = np.hstack((Z, -Z))
                else:
                    D = np.hstack((W, -W))
            elif dirgen == 1:
                # Random sample of the previous one
                if not preslineq and not presbounds:
                    nbdir = ceil(2*m*p0)
                    D = randcolZsearch(Z, np.arange(m), np.arange(m), nbdir)
                else:
                    nbdir = ceil(2*ni*p0)
                    D = randcolZsearch(W, np.arange(ni), np.arange(ni), nbdir)

            elif dirgen == 2:
                # Using uniform distribution in the null space of the equality constraints
                D = 2 * np.random.random((n, max(1, ceil(nbunc / 2)))) - 1
                if not preslineq and not presbounds:
                    D = Z.T @ D
                else:
                    D = W.T @ D

                for i in range(D.shape[1]):
                    D[:, i] /= np.linalg.norm(D[:, i])

                if not preslineq and not presbounds:
                    D = Z @ D
                else:
                    D = W @ D

                D = np.hstack((D, -D))
            else:
                raise RuntimeError("dirgen value not supported")

        else:
            # Case nbgen >=1 :
            if preslineq or (presleq and presbounds):
                Ys, Yc, calldd = gentech(Aeq, Ai, Z, Iap, Iam)
                if calldd:
                    sumdd += 1

                # Orthonormalization process
                if Ys.size > 0:
                    for i in range(Ys.shape[1]):
                        Ys[:, i] /= np.linalg.norm(Ys[:, i])

                    if Yc.size > 0:
                        Yc = Yc - Ys @ Ys.T @ Yc
                        for i in range(Yc.shape[1]):
                            Yc[:, i] /= np.linalg.norm(Yc[:, i])

                # Additional directions - currently empty
    			# Possibilities: coordinate vectors, more generators, etc.
                Ya = np.array((n, 0), dtype=float)

            else:
                # Simply select the complement of the active coordinate vectors
                Is = np.intersect1d(Iip, Iim)
                Icp = np.setdiff1d(Iip, Is)
                Icm = np.setdiff1d(Iim, Is)
                Ys = W[:, Is]
                Yc = np.hstack((W[:, Icp], -W[:, Icm]))

            dyy = 2*Ys.shape[1] + Yc.shape[1]

            if dyy > 0:
                if dirgen == 0:
                    D = np.hstack((Ys, -Ys, Yc))
                    if D.size > 0:
                        pDD = np.random.permutation(D.shape[1])
                        D = D[:, pDD]
                elif dirgen == 1:
                    # Ensuring a p > p0 probability of using a feasible descent direction among the columns of Z and their opposite
                    nbdir = ceil(dyy*p0)
                    D = randcolZsearch(np.hstack((Ys, -Ys, Yc)), np.arange(dyy), [], nbdir)
                elif dirgen == 2:
                    # Using uniform distribution in subspaces if possible, in cones otherwise
                    D = np.array((n, 0), dtype=float)
                    # Subspace vectors
                    nbdir = ceil(nbgen*p0)

                    if Ys.size > 0:
                        Dsb = 2 * np.random.random((Ys.shape[1], max(1, ceil(nbunc / 2)))) - 1
                        Dsb = Ys @ Dsb
                        for i in range(Dsb.shape[1]):
                            Dsb[:, i] /= np.linalg.norm(Dsb[:, i])

                        D = np.hstack((Dsb, -Dsb))

                    # Cone vectors
                    if Yc.size > 0:
                        nyc = Yc.shape[1]
                        nbdir = ceil(nyc*p0)
                        Dco = randcolZsearch(Yc,np.arange(nyc), [], nbdir)
                        D = np.hstack((D, Dco))
                else:
                    raise RuntimeError('dirgen value not supported')
            else:
                D = np.array((n, 0), dtype=float)

            # Adding additional directions if any
            if Ya.size > 0:
                if dirgen > 0:
                    ca = Ya.shape[1]
                    dda = ceil(ca*p0)
                    Ya = Ya[:, dda]
                D = np.hstack((D, Ya))

        # Additional directions II - Infeasible constraints normals
        # Set the boolean addII to 1 to activate this option
        addII = False
        # TODO this is not implemented

        if D.size > 0:
            cardD = 0
            print("Warning: empty polling set on iteration %g" % numIter)
        else:
            cardD = D.shape[1]

        numDirgen += cardD

        # Loop on the directions
        count_dir = 1
        success = False
        oldnumFeaspts = numFeaspts
        while not success and count_dir <= cardD and funcCount < maxevals:
            d = D[:, count_dir]
            xtemp = x + alpha*d

            # Test for the feasibility of the point
            if presbounds:
                xl = (min(xtemp-lb) > -tol_feas)
                xu = (max(xtemp-ub) < tol_feas)
            else:
                xl = True
                xu = True

            if Aeq.size > 0:
                Ax = (np.linalg.norm(Aeq @ xtemp - beq, np.inf) < tol_feas)
            else:
                Ax = True

            if preslineq:
                xli = (np.min(Aineq @ xtemp-lbineq) > -tol_feas)
                xui = (np.max(Aineq @ xtemp-ubineq) < tol_feas)
                Aix = (xli and xui)
            else:
                Aix = True

            if not xl or not xu or not Ax or not Aix:
                # print('Unfeasible tentative point %g' % dirgen)
                count_dir += 1
            else:
                numFeaspts += 1

                # Test for sufficient decrease at xtemp
                ftemp = fun(xtemp)
                funcCount += 1
                if ftemp < f - rho*(np.linalg.norm(d)**2):
                    success = True
                else:
                    count_dir += 1

        # End of the iteration
        # Have we sufficiently improved the current function value?
        if success:
            f = ftemp
            x = xtemp
            alpha = min(gamma * alpha, alphamax)
        else:
            alpha = theta*alpha

        histF = np.vstack((histF, np.array([f, funcCount])))

        # Is the stopping criterion satisfied?
        stopcrit = ((f - ftarget < diff0) or (alpha < tol_alpha) or (funcCount >= maxevals))

    # END OF THE MAIN LOOP

    # Final adjustments
    xsol = x
    fsol = f
    output = {}
    output['numIter'] = numIter
    output['funcCount'] = funcCount
    output['meanFeaspts'] = numFeaspts // numIter
    output['meanDirgen'] = numDirgen // numIter
    output['fracDD'] = sumdd / numIter
    output['histF'] = histF

    if alpha >=tol_alpha:
        if funcCount == 1:
            # No initial feasible point was found
            exitflag = 3
        else:
            # Budget of function evaluations exceeded
            exitflag = 2
    else:
        if f-ftarget < diff0:
            exitflag = 0
        else:
            # Step size shrunk below tolerance
            exitflag = 1

    return xsol, fsol, exitflag, output


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
    print("Done")
    return


if __name__ == '__main__':
    main()
