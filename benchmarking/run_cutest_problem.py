"""
Basic script to run a single CUTEst problem (with bound/linear constraints)
"""
import numpy as np
import directsearch

from get_all_cutest_problems import read_json
from get_reduced_cutest_problems import get_cutest_problem, get_bounds, get_linear_cons

def load_problem(probset, idx, all_problem_info):
    if probset in all_problem_info:
        if 0 <= idx < len(all_problem_info[probset]):
            entry = all_problem_info[probset][idx]
            prob = get_cutest_problem(entry['name'], sifParams=entry['sifparams'])
            x0 = np.array(entry['x0'])
            return prob, x0
        else:
            raise RuntimeError("Invalid index %g for probset %s (len probset = %g)" % (idx, probset, len(all_problem_info[probset])))
    else:
        raise RuntimeError("Unknown probset '%s'" % probset)


def solve_cutest_problem(prob, x0, poll_normal_cone=True):
    A_bounds, b_bounds = get_bounds(prob)  # bounds in the form A_bounds @ x <= b_bounds
    A_ineq, b_ineq, Aeq, beq = get_linear_cons(prob)  # linear constraints in the form A_ineq @ x <= b_ineq and Aeq @ x == b_eq

    assert len(beq) == 0, "directsearch cannot handle linear equality constraints currently"

    # Concatenate bounds and linear inequality constraints into A @ x <= b
    A = np.vstack((A_bounds, A_ineq))
    b = np.concatenate((b_bounds, b_ineq))

    soln, iter_counts = directsearch.solve(prob.obj, x0, A=A, b=b,
                                           return_iteration_counts=True,
                                           poll_normal_cone=poll_normal_cone)
    return soln, iter_counts


def main():
    # BOUNDS_ONLY has 93 problems, HAS_LINCONS has 45 problems (max idx = 92 or 44)
    # probset, idx = 'BOUNDS_ONLY', 4  # 0 <= idx < 93
    probset, idx = 'HAS_LINCONS', 0  # 0 <= idx < 45

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
