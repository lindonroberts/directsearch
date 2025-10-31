"""
Basic script to run a single CUTEst problem (with bound/linear constraints)
"""
import numpy as np
import directsearch

from get_all_cutest_problems import read_json
from get_reduced_cutest_problems import get_cutest_problem, get_bounds, get_linear_cons
from run_all_cutest import load_problem, ObjfunWrapper


def solve_cutest_problem(prob, x0, poll_normal_cone=True):
    objfun = ObjfunWrapper(prob, x0, print_all_evals=True)
    objfun.clear()
    print("x0 =", objfun.x0)
    print("A =", objfun.A)
    print("b =", objfun.b)
    soln, iter_counts, info = directsearch.solve(objfun, objfun.x0, A=objfun.A, b=objfun.b,
                                           maxevals=200 * (objfun.prob.n + 1),
                                           return_iteration_counts=True,
                                                 rho_uses_normd=False,
                                           poll_normal_cone=poll_normal_cone, verbose=False,
                                                 detailed_info_lincons = True, true_gradf = lambda x: prob.grad(x))
    return soln, iter_counts


def main():
    # BOUNDS_ONLY has 93 problems, HAS_LINCONS has 45 problems (max idx = 92 or 44)
    # probset, idx = 'BOUNDS_ONLY', 11  # 0 <= idx < 93
    probset, idx = 'HAS_LINCONS', 19  # 0 <= idx < 45

    all_problem_info = read_json('cutest_problems_reduced.json')
    prob, x0 = load_problem(probset, idx, all_problem_info)
    print("Solving for problem %s with n=%g" % (prob.name, prob.n))
    soln, iter_counts = solve_cutest_problem(prob, x0, poll_normal_cone=True)
    print(soln)
    print(iter_counts)

    print("Done")
    return


if __name__ == '__main__':
    main()
