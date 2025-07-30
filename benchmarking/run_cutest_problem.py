"""
Basic script to run a single CUTEst problem (with bound/linear constraints)
"""
import numpy as np
import pycutest
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
    if prob.m > 0:  # has (linear) constraints
        nlineq = np.logical_and(prob.is_eq_cons, prob.is_linear_cons).sum()
        nlinineq = np.logical_and(np.logical_not(prob.is_eq_cons), prob.is_linear_cons).sum()
    else:
        nlineq = 0
        nlinineq = 0

    A = np.zeros((nlinineq, prob.n), dtype=float)
    b = np.zeros((nlinineq,), dtype=float)
    Aeq = np.zeros((nlineq, prob.n), dtype=float)
    beq = np.zeros((nlineq,), dtype=float)

    if nlinineq > 0:
        idx_linineq = np.where(np.logical_and(np.logical_not(prob.is_eq_cons), prob.is_linear_cons))[0]
        for i in idx_linineq:
            g = prob.grad(prob.x0, i)  # gradient of i-th (linear) constraint
            if prob.cl[i] > PYCUTEST_NEG_INF:
                # g @ x >= cl[i] --> (-g) @ x <= -cl[i]
                pass
            if prob.cu[i] < PYCUTEST_INF:
                # g @ x <= cu[i]
                pass
        raise RuntimeError("Linear inequality extraction not implemented yet")

    if nlineq > 0:
        idx_lineq = np.where(np.logical_and(prob.is_eq_cons, prob.is_linear_cons))[0]
        raise RuntimeError("Linear equality extraction not implemented yet")

    return A, b, Aeq, beq


def main():
    # idx = 0  # bounds only
    # idx = 4  # bounds and linear inequalities
    idx = 5  # everything
    prob = get_problem(idx)
    print(prob)

    # Bounds
    A_bd, b_bd = get_bounds(prob)
    nbounds = len(b_bd)
    print("-----")
    print("Problem has %g bounds (Ax <= b)" % nbounds)
    print(A_bd)
    print(b_bd)

    # Linear constraints
    A, b, Aeq, beq = get_linear_cons(prob)
    nlinineq = len(b)
    nlineq = len(beq)
    # print("-----")
    # print("Problem has %g linear inequality constraints (Ax <= b)" % nlinineq)
    # print(A)
    # print(b)
    # print("-----")
    # print("Problem has %g linear equality constraints (Ax = b)" % nlineq)
    # print(Aeq)
    # print(beq)
    # print("-----")
    print("Done")
    return


if __name__ == '__main__':
    main()
