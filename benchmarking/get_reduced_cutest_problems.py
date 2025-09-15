"""
get_all_cutest_problems.py finds *all* of the test problems from
    Gratton, Royer, Vicente & Zhang. Direct search based on probabilistic feasible descent for bound and linearly
    constrained problems. Computational Optimization and Applications 72:3 (2019), pp. 525-559.

These are saved in cutest_problems.json.

Here, we process this list based on:
- Remove duplicates across problem sets
- Remove linear equality constraints (directsearch cannot handle this yet) - fixed variables can be automatically
    removed by PyCUTEst
- Calculate a feasible starting point (by projecting CUTEst default x0), as required by directsearch
"""
import numpy as np
import pycutest
from scipy.optimize import linprog, minimize, LinearConstraint

from get_all_cutest_problems import write_json, read_json

PYCUTEST_INF = 1e20
PYCUTEST_NEG_INF = -1e20

def get_cutest_problem(probname, sifParams=None):
    return pycutest.import_problem(probname, sifParams=sifParams, drop_fixed_variables=True)


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
        finite_cons = np.logical_or(prob.cl > PYCUTEST_NEG_INF, prob.cu < PYCUTEST_INF)
        idx_lineq = np.where(np.logical_and(finite_cons, np.logical_and(prob.is_eq_cons, prob.is_linear_cons)))[0]
        idx_linineq = np.where(np.logical_and(finite_cons, np.logical_and(np.logical_not(prob.is_eq_cons), prob.is_linear_cons)))[0]
    else:
        idx_lineq = []
        idx_linineq = []

    nlinineq = 0
    for i in idx_linineq:
        if prob.cl[i] > PYCUTEST_NEG_INF:
            nlinineq += 1
        if prob.cu[i] < PYCUTEST_INF:
            nlinineq += 1

    nlineq = 0
    for i in idx_lineq:
        if prob.cu[i] < PYCUTEST_INF:  # cl[i] = cu[i], no need to check both
            nlineq += 1

    A = np.zeros((nlinineq, prob.n), dtype=float)
    b = np.zeros((nlinineq,), dtype=float)
    Aeq = np.zeros((nlineq, prob.n), dtype=float)
    beq = np.zeros((nlineq,), dtype=float)

    if nlinineq > 0:
        idx = 0
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
        for i in idx_lineq:
            c, g = prob.cons(prob.x0, index=i, gradient=True)  # c + g @ (x-x0) = cu
            c -= np.dot(g, prob.x0)  # c + g @ x = cu
            if prob.cu[i] < PYCUTEST_INF:
                # g @ x = cu[i] - c
                Aeq[idx, :] = g
                beq[idx] = prob.cu[i] - c
                idx += 1

    return A, b, Aeq, beq


def feasibility(A, b, Aeq, beq, x):
    # Feasibility of x w.r.t. A @ x <= b and Aeq @ x = beq
    if b is not None and len(b) > 0:
        f_ineq = np.max(np.abs(np.maximum(A @ x - b, 0.0)))  # || max(A @ x - b, 0) ||_{infty}
    else:
        f_ineq = 0.0
    if beq is not None and len(beq) > 0:
        f_eq = np.max(np.abs(Aeq @ x - beq))  # ||Aeq @ x - beq||_{infty}
    else:
        f_eq = 0.0
    return max(f_ineq, f_eq, 0.0)


def find_feasible_point(A, b, Aeq, beq, x0):
    # Given linear constraints A @ x <= b and Aeq @ x = beq, find a feasible point by projecting x0.
    # - If the projection fails, compute a default feasible point by solving an LP
    ZERO_THRESH = 10.0 * np.finfo(float).eps  # as used in nearby_constraints() in lincons.py
    if feasibility(A, b, Aeq, beq, x0) <= ZERO_THRESH:
        # Current x0 is feasible
        return x0
    # Following PRIMA (https://github.com/libprima/prima/blob/main/python/prima/_common.py), set up the projection
    # problem and solve using scipy.optimize.minimize
    lincons = []
    if len(b) > 0:
        lincons.append(LinearConstraint(A, ub=b))
    if len(beq) > 0:
        lincons.append(LinearConstraint(Aeq, lb=beq, ub=beq))
    soln = minimize(lambda x: 0.5 * np.dot(x - x0, x - x0), x0, jac=lambda x: (x - x0), constraints=lincons)
    x0_new = soln.x
    if feasibility(A, b, Aeq, beq, x0_new) <= ZERO_THRESH:
        return x0_new

    # Projection failed, try solving an LP
    n = A.shape[1]  # problem dimension
    res = linprog(np.zeros((n,), dtype=float), A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq, bounds=(None, None))
    x0_new = res.x
    if feasibility(A, b, Aeq, beq, x0_new) <= ZERO_THRESH:
        return x0_new
    else:
        return None


def reduce_problems(outfile):
    all_problems = read_json('cutest_problems.json')
    reduced_problems = {}
    reduced_problems['BOUNDS_ONLY'] = []
    reduced_problems['HAS_LINCONS'] = []

    for problem_set in all_problems:
        print("*** %s ***" % problem_set)
        print("{0:^10}{1:^5}{2:^5}{3:^5}{4:^5}".format('NAME', 'N', 'NB', 'LE', 'LI'))
        for p in all_problems[problem_set]:
            probname = p['name']
            sifparams = p['sifparams']
            prob = get_cutest_problem(probname, sifParams=sifparams)
            A_bounds, b_bounds = get_bounds(prob)  # bounds in the form A_bounds @ x <= b_bounds
            A_ineq, b_ineq, Aeq, beq = get_linear_cons(
                prob)  # linear constraints in the form A_ineq @ x <= b_ineq and Aeq @ x == b_eq
            nbounds = len(b_bounds)
            nlinineq = len(b_ineq)
            nlineq = len(beq)

            # Drop any problem with linear equality constraints
            if nlineq > 0:
                print(" - Skipping %s since has linear equality constraints" % probname)
                continue

            # Check for duplicates in current set of reduced_problems
            probset = 'HAS_LINCONS' if nlinineq > 0 else 'BOUNDS_ONLY'
            for entry in reduced_problems[probset]:
                if probname == entry['name']:
                    # print("Possible duplicate %s" % probname)
                    if sifparams == entry['sifparams']:
                        print(" - Skipping duplicate problem %s (sifParams = %s)" % (probname, str(sifparams)))
                        continue
                    # if sifparams == reduced_problems[probset][probname]['sifparams']:
                    #     print(" - Skipping duplicate problem %s (sifParams = %s)" % (probname, str(sifparams)))
                    #     continue

            # Calculate feasible x0
            A = np.vstack((A_bounds, A_ineq))
            b = np.concatenate((b_bounds, b_ineq))
            x0 = find_feasible_point(A, b, Aeq, beq, prob.x0)
            if x0 is None:
                print(" - Skipping %s - could not find feasible point" % probname)
                continue

            this_prob = {}
            this_prob['name'] = probname
            this_prob['sifparams'] = sifparams
            this_prob['n'] = prob.n
            this_prob['nbounds'] = nbounds
            this_prob['nlinineq'] = nlinineq
            this_prob['nlineq'] = nlineq
            this_prob['x0'] = [float(xi) for xi in x0]
            reduced_problems[probset].append(this_prob)
            print("{0:^10}{1:^5}{2:^5}{3:^5}{4:^5}".format(probname, prob.n, nbounds, nlineq, nlinineq))

    write_json(reduced_problems, outfile)
    return


def get_feasible_x0_demo():
    # Simple demo of finding feasible point via projection
    # see_all = True  # see all projection results
    see_all = False  # see details for specific problems

    selected_problems = ['ALLINIT', 'CAMEL6', 'BT3', 'HS48', 'HS55', 'AVGASA', 'HS24', 'BIGGSC4']

    all_problems = read_json('cutest_problems.json')
    if see_all:
        for problem_set in all_problems:
            print("*** %s ***" % problem_set)
            print("{0:^10}{1:^5}{2:^5}{3:^5}{4:^5}".format('NAME', 'N', 'NB', 'LE', 'LI'))
            for p in all_problems[problem_set]:
                probname = p['name']
                sifparams = p['sifparams']
                prob = get_cutest_problem(probname, sifParams=sifparams)
                A_bounds, b_bounds = get_bounds(prob)  # bounds in the form A_bounds @ x <= b_bounds
                A_ineq, b_ineq, Aeq, beq = get_linear_cons(
                    prob)  # linear constraints in the form A_ineq @ x <= b_ineq and Aeq @ x == b_eq
                nbounds = len(b_bounds)
                nlinineq = len(b_ineq)
                nlineq = len(beq)

                # Calculate feasible x0
                A = np.vstack((A_bounds, A_ineq))
                b = np.concatenate((b_bounds, b_ineq))
                x0 = find_feasible_point(A, b, Aeq, beq, prob.x0)

                print("{0:^10}{1:^5}{2:^5}{3:^5}{4:^5}".format(probname, prob.n, nbounds, nlineq, nlinineq))
                if x0 is not None:
                    print(np.linalg.norm(x0 - prob.x0))
                else:
                    print("Could not find x0")

    else:
        for problem_set in all_problems:
            for p in all_problems[problem_set]:
                probname = p['name']
                if probname not in selected_problems:
                    continue

                sifparams = p['sifparams']
                prob = get_cutest_problem(probname, sifParams=sifparams)
                A_bounds, b_bounds = get_bounds(prob)  # bounds in the form A_bounds @ x <= b_bounds
                A_ineq, b_ineq, Aeq, beq = get_linear_cons(
                    prob)  # linear constraints in the form A_ineq @ x <= b_ineq and Aeq @ x == b_eq
                nbounds = len(b_bounds)
                nlinineq = len(b_ineq)
                nlineq = len(beq)

                # Calculate feasible x0, A @ x <= b and Aeq @ x = beq
                A = np.vstack((A_bounds, A_ineq))
                b = np.concatenate((b_bounds, b_ineq))
                x0 = find_feasible_point(A, b, Aeq, beq, prob.x0)

                print("*** %s ***" % prob.name)
                if len(b) > 0 and len(beq) > 0:
                    # Ax <= b --> -A*x >= -b
                    Afull = np.vstack((-A, Aeq, -Aeq))  # Aeq*x == beq --> Aeq*x >= beq and -Aeq*x >= -beq
                    bfull = np.concatenate((-b, beq, -beq))
                elif len(b) > 0:
                    # LI only
                    Afull = -A
                    bfull = -b
                else:
                    # LE only
                    Afull = np.vstack((Aeq, -Aeq))
                    bfull = np.concatenate((beq, -beq))

                print("A.shape =", Afull.shape)
                m, n = Afull.shape
                for i in range(m):
                    this_line = ""
                    for j in range(n):
                        this_line += "A[%g*n+%g] = %g; " % (i, j, Afull[i,j])
                    print(this_line)

                this_line = ""
                for i in range(m):
                    this_line += "b[%g] = %g; " % (i, bfull[i])
                    if (i+1) % 5 == 0:
                        print(this_line)
                        this_line = ""
                print(this_line)

                this_line = ""
                for i in range(n):
                    this_line += "x0[%g] = %g; " % (i, prob.x0[i])
                    if (i + 1) % 5 == 0:
                        print(this_line)
                        this_line = ""
                print(this_line)

                this_line = ""
                for i in range(n):
                    this_line += "xtrue[%g] = %.17g; " % (i, x0[i])
                    if (i + 1) % 5 == 0:
                        print(this_line)
                        this_line = ""
                print(this_line)
                print("")
    return

def main():
    if True:
        get_feasible_x0_demo()
        return

    reduced_problems_file = 'cutest_problems_reduced.json'
    if False:
        reduce_problems(reduced_problems_file)

    # Print statistics
    all_problems = read_json(reduced_problems_file)
    for problem_set in all_problems:
        print("*** %s (%g probs) ***" % (problem_set, len(all_problems[problem_set])))
        print("{0:^10}{1:^5}{2:^5}{3:^5}{4:^5}{5:^15}".format('NAME', 'N', 'NB', 'LE', 'LI', 'X0_FEAS'))
        for p in all_problems[problem_set]:
            probname = p['name']
            sifparams = p['sifparams']
            prob = get_cutest_problem(probname, sifParams=sifparams)
            A_bounds, b_bounds = get_bounds(prob)  # bounds in the form A_bounds @ x <= b_bounds
            A_ineq, b_ineq, Aeq, beq = get_linear_cons(prob)  # linear constraints in the form A_ineq @ x <= b_ineq and Aeq @ x == b_eq
            nbounds = len(b_bounds)
            nlinineq = len(b_ineq)
            nlineq = len(beq)

            # Calculate feasible x0
            A = np.vstack((A_bounds, A_ineq))
            b = np.concatenate((b_bounds, b_ineq))
            x0_feas = feasibility(A, b, Aeq, beq, np.array(p['x0']))

            print("{0:^10}{1:^5}{2:^5}{3:^5}{4:^5}{5:^15.4e}".format(probname, prob.n, nbounds, nlineq, nlinineq, x0_feas))
    print("Done")
    return

if __name__ == '__main__':
    main()
