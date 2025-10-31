"""
Run all CUTEst problems from cutest_problems_reduced.json
"""
import datetime
import numpy as np
import os
import time

import directsearch

from get_all_cutest_problems import read_json, write_json
from get_reduced_cutest_problems import get_cutest_problem, get_bounds, get_linear_cons, feasibility

class ObjfunWrapper(object):
    def __init__(self, prob, x0, print_all_evals=False):
        self.prob = prob
        self.x0 = x0
        self.print_all_evals = print_all_evals

        A_bounds, b_bounds = get_bounds(prob)  # bounds in the form A_bounds @ x <= b_bounds
        A_ineq, b_ineq, Aeq, beq = get_linear_cons(prob)  # linear constraints in the form A_ineq @ x <= b_ineq and Aeq @ x == b_eq
        self.nbounds = len(b_bounds)
        self.nlineq = len(beq)
        self.nlinineq = len(b_ineq)
        assert len(beq) == 0, "directsearch cannot handle linear equality constraints currently"

        # All bound and linear constraints in the form A @ x <= b, for use in directsearch
        self.A = np.vstack((A_bounds, A_ineq))
        self.b = np.concatenate((b_bounds, b_ineq))

        # Evaluation history to save
        self._f_history = []
        self._maxcv_history = []
        self.clear()

    def clear(self):
        self._f_history = []
        self._maxcv_history = []

    def get_history(self, as_numpy=True):
        if as_numpy:
            return np.array(self._f_history), np.array(self._maxcv_history)
        else:
            return self._f_history, self._maxcv_history

    def __call__(self, x):
        fx = self.prob.obj(x)
        maxcv = feasibility(self.A, self.b, None, None, x)
        self._f_history.append(fx)
        self._maxcv_history.append(maxcv)
        if self.print_all_evals:
            print(x, "fx = %g, maxcv = %g" % (fx, maxcv))
        return fx


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

def solve_all_problems(run_name, budget_in_gradients=200, skip_existing=True, lincons_only=False):
    # All directsearch.solve() settings for a given run should be put here
    if run_name == 'tangent_only':
        poll_normal_cone = False
        rho_uses_normd = True  # default choice
        get_detailed_info = False  # no valid Lambda here
    elif run_name == 'tangent_only_simple_rho':
        poll_normal_cone = False
        rho_uses_normd = False  # to match with tangent_and_normal
        get_detailed_info = False  # no valid Lambda here
    elif run_name.startswith('tangent_and_normal'):
        poll_normal_cone = True
        rho_uses_normd = False  # required when poll_normal_cone=True
        get_detailed_info = run_name.endswith('detailed')
    else:
        raise RuntimeError("Unknown run_name '%s'" % run_name)

    outdir = os.path.join('raw_results', run_name)
    os.makedirs(outdir, exist_ok=True)

    #####
    all_problem_info = read_json('cutest_problems_reduced.json')
    for probset in all_problem_info:
        if lincons_only and probset != 'HAS_LINCONS':
            print("*** SKIPPING PROBLEM SET %s ***" % probset)
            continue
        print("*** %s ***" % probset)
        for idx in range(len(all_problem_info[probset])):
            prob, x0 = load_problem(probset, idx, all_problem_info)
            true_gradf = lambda x: prob.grad(x)
            objfun = ObjfunWrapper(prob, x0)
            maxfun = budget_in_gradients * (prob.n + 1)
            outfile = '%s_%g_%s.json' % (probset, idx, prob.name)
            if skip_existing and os.path.isfile(os.path.join(outdir, outfile)):
                print(" - Skipping existing run for %s (n=%g)" % (prob.name, prob.n))
                continue

            this_results = {}
            this_results['probset'] = probset
            this_results['probset_idx'] = idx
            this_results['name'] = prob.name
            this_results['sifparams'] = prob.sifParams
            this_results['n'] = prob.n
            this_results['nbounds'] = objfun.nbounds
            this_results['nlinineq'] = objfun.nlinineq
            this_results['nlineq'] = objfun.nlineq

            this_results['budget_in_gradients'] = budget_in_gradients
            this_results['maxfun'] = maxfun
            this_results['poll_normal_cone'] = poll_normal_cone
            this_results['rho_uses_normd'] = rho_uses_normd

            print("- [idx=%g] %s (n=%g)" % (idx, prob.name, prob.n))
            objfun.clear()
            try:
                if get_detailed_info:
                    start_time, start_cpu_clock = datetime.datetime.now(), time.process_time()
                    soln, iter_counts, info = directsearch.solve(objfun, objfun.x0, maxevals=maxfun,
                                                                 A=objfun.A, b=objfun.b,
                                                                 return_iteration_counts=True,
                                                                 rho_uses_normd=rho_uses_normd,
                                                                 poll_normal_cone=poll_normal_cone,
                                                                 detailed_info_lincons=True,
                                                                 true_gradf=true_gradf)
                    stop_time, stop_cpu_clock = datetime.datetime.now(), time.process_time()
                else:
                    start_time, start_cpu_clock = datetime.datetime.now(), time.process_time()
                    soln, iter_counts = directsearch.solve(objfun, objfun.x0, maxevals=maxfun,
                                                           A=objfun.A, b=objfun.b,
                                                           return_iteration_counts=True,
                                                           rho_uses_normd=rho_uses_normd,
                                                           poll_normal_cone=poll_normal_cone,
                                                           detailed_info_lincons=False)
                    stop_time, stop_cpu_clock = datetime.datetime.now(), time.process_time()
                    info = None
            except Exception as e:
                print("*** ERROR, check later -- ", str(e))
                # exit()
                continue
            # print(info['lambda_alpha_gradf'])
            # exit()  # stop after first problem for testing purposes

            this_results['run_start_time'] = start_time.strftime("%Y-%m-%d %H:%M:%S")
            this_results['run_stop_time'] = stop_time.strftime("%Y-%m-%d %H:%M:%S")
            this_results['run_wall_time'] = (stop_time - start_time).total_seconds()  # in seconds
            this_results['run_cpu_time'] = (stop_cpu_clock - start_cpu_clock)  # in seconds

            f_hist, maxcv_hist = objfun.get_history(as_numpy=False)
            this_results['results'] = {}
            this_results['results']['nf'] = len(f_hist)
            this_results['results']['niters_successful_tangent'] = iter_counts['successful']
            this_results['results']['niters_successful_normal'] = iter_counts['successful_negative_direction']
            this_results['results']['niters_unsuccessful'] = iter_counts['unsuccessful']
            this_results['results']['f_history'] = f_hist
            this_results['results']['maxcv_history'] = maxcv_hist
            if get_detailed_info:
                this_results['results']['detailed_info'] = info

            write_json(this_results, os.path.join(outdir, outfile))
    return


def main():
    budget_in_gradients = 200
    run_names = []
    # run_names.append('tangent_only')  # cannot get detailed info
    # run_names.append('tangent_only_simple_rho')  # cannot get detailed info
    # run_names.append('tangent_and_normal')  # used for main results, no detailed info
    run_names.append('tangent_and_normal_detailed')  # new run with detailed info

    # skip_existing = True  # new runs only
    skip_existing = False  # overwrite old runs

    # lincons_only = False  # bounds and lincons
    lincons_only = True  # lincons only

    for run_name in run_names:
        print("")
        print("***********************")
        print("*** NEW RUN: %s ***" % run_name)
        print("***********************")
        solve_all_problems(run_name, budget_in_gradients=budget_in_gradients, skip_existing=skip_existing,
                           lincons_only=lincons_only)
    print("Done")
    return


if __name__ == "__main__":
    main()
