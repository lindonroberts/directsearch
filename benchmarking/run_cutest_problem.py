"""
Basic script to run a single CUTEst problem (with bound/linear constraints)
"""
import numpy as np
import pycutest
from scipy.optimize import linprog

import directsearch

PYCUTEST_INF = 1e20
PYCUTEST_NEG_INF = -1e20

def get_problem(idx, drop_fixed_variables=True):
    if idx == 0:
        print("Loading ALLININT - bound constraints only (n=4)")
        probname = 'ALLINIT'
        sifParams = None
    elif idx == 1:
        print("Loading BOOTH - linear equality constraints only (n=2)")
        probname = 'BOOTH'
        sifParams = None
    elif idx == 2:
        print("Loading HONG - bounds and linear equality constraints (n=4)")
        probname = 'HONG'
        sifParams = None
    elif idx == 3:
        print("Loading HS268 - linear inequality constraints only (n=5)")
        probname = 'HS268'
        sifParams = None
    elif idx == 4:
        print("Loading HS21 - bounds and linear inequality constraints (n=2)")
        probname = 'HS21'
        sifParams = None
    elif idx == 5:
        print("Loading DUALC1 - bounds, linear equality & linear inequality (n=9)")
        probname = 'DUALC1'
        sifParams = None
    else:
        raise RuntimeError("Unknown problem index %s" % str(idx))
    return pycutest.import_problem(probname, sifParams=sifParams, drop_fixed_variables=drop_fixed_variables)

def get_bounds(prob):
    # Extract (finite) problem bounds in linear inequality constraint form expected by directsearch, A @ x <= b
    # TODO Assumes no fixed variables

    # How many actual/finite bounds do we have?
    nbounds = (prob.bl > PYCUTEST_NEG_INF).sum() + (prob.bu < PYCUTEST_INF).sum()
    A = np.zeros((nbounds, prob.n), dtype=float)
    b = np.zeros((nbounds,), dtype=float)

    if nbounds == 0:
        return A, b

    idx = 0
    for i in range(prob.n):
        if prob.bl[i] > PYCUTEST_NEG_INF:
            # x[i] >= bl[i] --> (-ei) @ x <= -bl[i]
            A[idx, i] = -1.0
            b[idx] = -prob.bl[i]
            idx += 1
        if prob.bu[i] < PYCUTEST_INF:
            # x[i] <= bu[i] --> ei @ x <= bu[i]
            A[idx, i] = 1.0
            b[idx] = prob.bu[i]
            idx += 1

    return A, b


def get_linear_cons(prob):
    # Extract (finite) linear constraints in form expected by directsearch, A @ x <= b (and Aeq @ x = beq, not usable yet)
    # print("m = %g" % prob.m)
    # print("is_linear =", prob.is_linear_cons)
    # print("is_eq_cons =", prob.is_eq_cons)
    # print("cl =", prob.cl)
    # print("cu =", prob.cu)
    # print("x0 =", prob.x0)
    # print(prob.cons(prob.x0, index=0, gradient=True))
    if prob.m > 0:  # has (linear) constraints
        finite_cons = np.logical_or(prob.cl > PYCUTEST_NEG_INF, prob.cu < PYCUTEST_INF)
        nlineq = np.logical_and(finite_cons, np.logical_and(prob.is_eq_cons, prob.is_linear_cons)).sum()
        nlinineq = np.logical_and(finite_cons, np.logical_and(np.logical_not(prob.is_eq_cons), prob.is_linear_cons)).sum()
    else:
        nlineq = 0
        nlinineq = 0

    A = np.zeros((nlinineq, prob.n), dtype=float)
    b = np.zeros((nlinineq,), dtype=float)
    Aeq = np.zeros((nlineq, prob.n), dtype=float)
    beq = np.zeros((nlineq,), dtype=float)

    if nlinineq > 0:
        idx = 0
        idx_linineq = np.where(np.logical_and(np.logical_not(prob.is_eq_cons), prob.is_linear_cons))[0]
        for i in idx_linineq:
            c, g = prob.cons(prob.x0, index=i, gradient=True)  # cl <= c + g @ (x-x0) <= cu
            c -= np.dot(g, prob.x0)  # cl <= c + g @ x <= cu
            if prob.cl[i] > PYCUTEST_NEG_INF:
                # c + g @ x >= cl[i] --> (-g) @ x <= c - cl[i]
                A[idx,:] = -g
                b[idx] = c - prob.cl[i]
                idx += 1
            if prob.cu[i] < PYCUTEST_INF:
                # g @ x <= cu[i] - c
                A[idx, :] = g
                b[idx] = prob.cu[i] - c
                idx += 1

    if nlineq > 0:
        idx = 0
        idx_lineq = np.where(np.logical_and(prob.is_eq_cons, prob.is_linear_cons))[0]
        for i in idx_lineq:
            c, g = prob.cons(prob.x0, index=i, gradient=True)  # c + g @ (x-x0) = cu
            c -= np.dot(g, prob.x0)  # c + g @ x = cu
            if prob.cu[i] < PYCUTEST_INF:
                # g @ x = cu[i] - c
                Aeq[idx, :] = g
                beq[idx] = prob.cu[i] - c
                idx += 1

    return A, b, Aeq, beq


def find_feasible_point(A, b, Aeq, beq):
    # Given linear constraints A @ x <= b and Aeq @ x = beq, find any feasible point (by solving an LP)
    # Required to set suitable x0, as directsearch requires feasible initial point
    n = A.shape[1]  # problem dimension
    res = linprog(np.zeros((n,), dtype=float), A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq, bounds=(None, None))
    if not res.success:
        raise RuntimeError("Failed to find feasible point")
    return res.x

def prepare_problem(prob):
    # Bounds
    A_bd, b_bd = get_bounds(prob)
    nbounds = len(b_bd)
    # print("-----")
    # print("Problem has %g bounds (A_bd @ x <= b_bd)" % nbounds)
    # print(A_bd)
    # print(b_bd)

    # Linear constraints (in the form A @ x <= b and Aeq @ x = beq)
    A, b, Aeq, beq = get_linear_cons(prob)
    nlinineq = len(b)
    nlineq = len(beq)
    if nlineq > 0:
        raise RuntimeError("directsearch cannot handle linear equality constraints currently")
    # print("-----")
    # print("Problem has %g linear inequality constraints (A @ x <= b)" % nlinineq)
    # print(A)
    # print(b)
    # print("-----")
    # print("Problem has %g linear equality constraints (Aeq @ x = beq)" % nlineq)
    # print(Aeq)
    # print(beq)
    # print("-----")

    # Stack bounds and linear inequalities
    Aineq = np.vstack((A_bd, A))
    bineq = np.concatenate((b_bd, b))
    # print(Aineq)
    # print(bineq)

    x0 = find_feasible_point(Aineq, bineq, Aeq, beq)
    return x0, Aineq, bineq

def main():
    # idx = 0  # bounds only
    # idx = 2  # bounds and linear equalities
    idx = 4  # bounds and linear inequalities
    # idx = 5  # everything
    prob = get_problem(idx)
    print(prob)

    x0, Aineq, bineq = prepare_problem(prob)
    soln, iter_counts = directsearch.solve(prob.obj, x0, A=Aineq, b=bineq, return_iteration_counts=True)
    print(soln)
    print(iter_counts)

    print("Done")
    return


if __name__ == '__main__':
    main()
