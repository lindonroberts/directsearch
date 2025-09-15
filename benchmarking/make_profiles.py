"""
Make data and performance profiles
"""
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from get_all_cutest_problems import write_json, read_json


MAXCV_THRESH = 1e-12  # exclude any evaluations with maxcv > MAXCV_THRESH (expect none)


def load_all_results_single_run(run_name, probsets):
    infolder = os.path.join('raw_results', run_name)
    all_problem_info = read_json('cutest_problems_reduced.json')
    single_run_results = {}
    for probset in probsets:
        print("Loading results for run %s and probset %s" % (run_name, probset))
        for idx in range(len(all_problem_info[probset])):
            entry = all_problem_info[probset][idx]
            infile = '%s_%g_%s.json' % (probset, idx, entry['name'])
            if os.path.isfile(os.path.join(infolder, infile)):
                this_problem_results = read_json(os.path.join(infolder, infile))
                single_run_results[infile.replace('.json', '')] = this_problem_results
            else:
                print(' - Skipping missing results: %s' % infile.replace('.json', ''))
    return single_run_results


def load_all_results(run_names, probsets, drop_missing_results=False):
    all_results = {}
    for run_name in run_names:
        all_results[run_name] = load_all_results_single_run(run_name, probsets)

    if drop_missing_results:
        # Some problems aren't working yet, so for now just exclude those problems where we don't have results
        # for all runs
        print("Dropping missing results")
        probcount = {}
        for run_name in run_names:
            for probkey in all_results[run_name]:
                if probkey in probcount:
                    probcount[probkey] += 1
                else:
                    probcount[probkey] = 1

        for probkey in probcount:
            if probcount[probkey] < len(run_names):
                for run_name in run_names:
                    if probkey in all_results[run_name]:
                        print("- Removing %s from run %s" % (probkey, run_name))
                        del all_results[run_name][probkey]

    for run_name in run_names:
        print("%s has results for %g problems" % (run_name, len(all_results[run_name])))
    return all_results


def get_f0_fmin(all_results):
    all_f0_fmin = {}
    probkeys = list(all_results[list(all_results.keys())[0]].keys())
    for probkey in probkeys:
        all_f0s = []
        all_fmins = []
        for run_name in all_results:
            f_hist = np.array(all_results[run_name][probkey]['results']['f_history'])
            maxcv_hist = np.array(all_results[run_name][probkey]['results']['maxcv_history'])
            f_hist[maxcv_hist > MAXCV_THRESH] = np.inf  # ignore infeasible points
            this_f0 = f_hist[0]  # x0 chosen to be feasible
            this_fmin = np.min(f_hist)
            all_f0s.append(this_f0)
            all_fmins.append(this_fmin)

        all_f0_fmin[probkey] = {'f0': float(np.median(all_f0s)), 'fmin': float(np.min(all_fmins))}
    return all_f0_fmin



def get_solve_times(all_results, tau_levels):
    """
    Extract solve times for each tau level

    Input all_results = dict as output from load_all_results()
    Input tau_levels = list of integers, e.g. 2 --> tau=1e-2

    Output: pandas dataframe with columns:
    - run_name = run name
    - probname = problem name
    - n = dimension
    - NB = number bounds
    - LI = number linear inequality constraints
    - LE = number linear equality constraints
    - maxevals = max evaluations for problem
    - f0, fmin = f0 and fmin value for this problem (across all solvers)
    - this_nf = number of evaluations this run used
    - this_fmin = fmin for this solver
    - tau1, tau2, ... = number of evaluations taken to achieve accuracy level tau=1e-1, 1e-2, ...
    """
    all_f0_fmin = get_f0_fmin(all_results)

    processed_results = {'run_name': [], 'probset': [], 'probname': [], 'n': [], 'NB': [], 'LI': [], 'LE': [],
                         'maxevals': [], 'f0': [], 'fmin': [],
                         'this_nf': [], 'this_fmin': []}
    for tau_int in tau_levels:
        processed_results['tau%g' % tau_int] = []
    for run_name in all_results:
        for probkey in all_results[run_name]:
            f0 = all_f0_fmin[probkey]['f0']
            fmin = all_f0_fmin[probkey]['fmin']

            this_results = all_results[run_name][probkey]
            probset = this_results['probset']
            n = this_results['n']
            NB = this_results['nbounds']
            LI = this_results['nlinineq']
            LE = this_results['nlineq']
            maxevals = this_results['maxfun']


            f_hist = np.array(all_results[run_name][probkey]['results']['f_history'])
            maxcv_hist = np.array(all_results[run_name][probkey]['results']['maxcv_history'])
            f_hist[maxcv_hist > MAXCV_THRESH] = np.inf  # ignore infeasible points
            this_fmin = np.min(f_hist)
            this_nf = len(f_hist)

            # Save all results
            processed_results['run_name'].append(run_name)
            processed_results['probset'].append(probset)
            processed_results['probname'].append(probkey)
            processed_results['n'].append(n)
            processed_results['NB'].append(NB)
            processed_results['LI'].append(LI)
            processed_results['LE'].append(LE)
            processed_results['maxevals'].append(maxevals)
            processed_results['f0'].append(f0)
            processed_results['fmin'].append(fmin)
            processed_results['this_nf'].append(this_nf)
            processed_results['this_fmin'].append(this_fmin)

            for tau_int in tau_levels:
                tau = 10 ** (-tau_int)  # tau_level=2 --> tau=1e-2
                fthresh = fmin + tau*(f0 - fmin)
                if fthresh >= this_fmin:
                    nf_solved = np.argmax(f_hist <= fthresh) + 1  # np.argmax(arr<=val) returns first index where true
                else:
                    nf_solved = -1
                processed_results['tau%g' % tau_int].append(nf_solved)

    return pd.DataFrame.from_dict(processed_results)



def build_data_profile_curves(processed_results, budget_in_gradients, tau_level, nvals=100):
    all_solvers = sorted(processed_results['run_name'].unique())
    all_problems = sorted(processed_results['probname'].unique())
    nproblems = len(all_problems)

    # Make data profiles
    data_profile = {}
    xvals = np.linspace(0.0, budget_in_gradients, nvals+1)
    data_profile['xvals'] = xvals
    nvals = len(xvals)

    for solver in all_solvers:
        this_solver_results = processed_results.loc[processed_results['run_name'] == solver]
        if len(this_solver_results) != nproblems:
            raise RuntimeError("Expect %g rows for solver %s, got %g rows" % (nproblems, solver, len(this_solver_results)))
        col = 'tau%g' % tau_level
        solved_budget = this_solver_results[col] / (this_solver_results['n'] + 1)
        solved_budget[this_solver_results[col] < 0] = -1.0

        # Build data profile curve (average over all runs)
        dp = np.zeros((nvals,))
        for i, budget in enumerate(xvals):
            nsolved = len(solved_budget[(solved_budget >= 0) & (solved_budget <= budget)])
            dp[i] = float(nsolved) / float(len(solved_budget))
        data_profile[solver] = dp

    return data_profile


def get_min_solve_times(processed_results, tau_level):
    # Minimum solve times for each problem (to build performance profiles)
    all_problems = sorted(processed_results['probname'].unique())
    min_solve_times = {}
    for probname in all_problems:
        this_problem_results = processed_results.loc[processed_results['probname'] == probname]
        col = 'tau%g' % tau_level
        # min solve time is smallest positive value
        if this_problem_results[col].max() > 0:
            min_solve_time = this_problem_results.loc[this_problem_results[col] > 0][col].min()
        else:
            min_solve_time = -1
        min_solve_times[probname] = int(min_solve_time)
    return min_solve_times


def build_perf_profile_curves(processed_results, tau_level, nvals=100, max_ratio=32.0):
    all_solvers = sorted(processed_results['run_name'].unique())
    all_problems = sorted(processed_results['probname'].unique())
    nproblems = len(all_problems)

    # Make data profiles
    perf_profile = {}
    xvals = 2.0 ** np.linspace(0.0, math.log(max_ratio, 2.0), nvals+1)
    perf_profile['xvals'] = xvals
    nvals = len(xvals)

    min_solve_times = get_min_solve_times(processed_results, tau_level)

    for solver in all_solvers:
        this_solver_results = processed_results.loc[processed_results['run_name'] == solver].copy()
        # Make a new column with min_solve_times[probname]
        this_solver_results['min_solve_time'] = this_solver_results['probname'].map(min_solve_times)
        if len(this_solver_results) != nproblems:
            raise RuntimeError("Expect %g rows for run_name %s, got %g rows" % (nproblems, solver, len(this_solver_results)))
        col = 'tau%g' % tau_level
        solved_ratio = this_solver_results[col] / this_solver_results['min_solve_time']
        solved_ratio[this_solver_results[col] < 0] = -1.0

        # Build data profile curve (average over all runs)
        pp = np.zeros((nvals,))
        for i, ratio in enumerate(xvals):
            nsolved = len(solved_ratio[(solved_ratio >= 0) & (solved_ratio <= ratio)])
            pp[i] = float(nsolved) / float(len(solved_ratio))
        perf_profile[solver] = pp

    return perf_profile


def plot_data_profile(data_profile_vals, plot_info, tau_level, fmt='eps', filestem=None, linewidth=2):
    if filestem is not None:
        font_size = 'large'  # x-large for presentations
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
    else:
        font_size = 'small'

    plt.figure()
    plt.clf()
    ax = plt.gca()  # current axes
    plot_fun = ax.plot
    xvals = data_profile_vals['xvals']

    results = {}
    results['xval'] = xvals
    for solver_plot_dict in plot_info:
        solver = solver_plot_dict['run_name']
        if solver not in data_profile_vals:
            print("(skipping solver %s, not used for this test set)" % solver)
            continue
        col = solver_plot_dict['col']
        ls = solver_plot_dict['ls']
        lbl = solver_plot_dict['lbl']
        mkr = solver_plot_dict['mkr']
        ms = solver_plot_dict['ms']
        dp = data_profile_vals[solver]
        if mkr != '':
            # If using a marker, only put the marker on a subset of points (to avoid cluttering)
            skip_array = np.mod(np.arange(len(dp)), len(dp) // 10) == 0
            # Line 1: the subset of points with markers
            plot_fun(xvals[skip_array], dp[skip_array], label='_nolegend_', color=col, linestyle='', marker=mkr, markersize=ms)
            # Line 2: a single point with the correct format, so the legend label can use this
            plot_fun(xvals[0], dp[0], label=lbl, color=col, linestyle=ls, marker=mkr, markersize=ms)
            # Line 3: the original line with no markers (or label)
            plot_fun(xvals, dp, label='_nolegend_', color=col, linestyle=ls, linewidth=linewidth, marker='', markersize=0)
        else:
            plot_fun(xvals, dp, label=lbl, color=col, linestyle=ls, linewidth=linewidth, marker='', markersize=0)
        results[lbl] = dp

    results_df = pd.DataFrame.from_dict(results)

    ax.set_xlabel(r"Budget in evals (gradients)", fontsize=font_size)
    ax.set_ylabel(r"Proportion problems solved", fontsize=font_size)

    ax.legend(loc='lower right', fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.axis([0, np.max(xvals), 0, 1])  # (xlow, xhigh, ylow, yhigh)
    ax.grid()

    if filestem is not None:
        results_df.to_csv("%s_data%g_raw.csv" % (filestem, tau_level), index=False)
        plt.savefig("%s_data%g.%s" % (filestem, tau_level, fmt), bbox_inches='tight')
    else:
        plt.show()
    return


def plot_perf_profile(perf_profile_vals, plot_info, tau_level, fmt='eps', filestem=None, linewidth=2):
    if filestem is not None:
        font_size = 'large'  # x-large for presentations
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
    else:
        font_size = 'small'

    plt.figure()
    plt.clf()
    ax = plt.gca()  # current axes
    plot_fun = ax.semilogx
    xvals = perf_profile_vals['xvals']

    results = {}
    results['xval'] = xvals
    for solver_plot_dict in plot_info:
        solver = solver_plot_dict['run_name']
        if solver not in perf_profile_vals:
            print("(skipping solver %s, not used for this test set)" % solver)
            continue
        col = solver_plot_dict['col']
        ls = solver_plot_dict['ls']
        lbl = solver_plot_dict['lbl']
        mkr = solver_plot_dict['mkr']
        ms = solver_plot_dict['ms']
        pp = perf_profile_vals[solver]
        if mkr != '':
            # If using a marker, only put the marker on a subset of points (to avoid cluttering)
            skip_array = np.mod(np.arange(len(pp)), len(pp)//10) == 0
            # Line 1: the subset of points with markers
            plot_fun(xvals[skip_array], pp[skip_array], label='_nolegend_', color=col, linestyle='', marker=mkr, markersize=ms)
            # Line 2: a single point with the correct format, so the legend label can use this
            plot_fun(xvals[0], pp[0], label=lbl, color=col, linestyle=ls, marker=mkr, markersize=ms)
            # Line 3: the original line with no markers (or label)
            plot_fun(xvals, pp, label='_nolegend_', color=col, linestyle=ls, linewidth=linewidth, marker='', markersize=0)
        else:
            plot_fun(xvals, pp, label=lbl, color=col, linestyle=ls, linewidth=linewidth, marker='', markersize=0)
        results[lbl] = pp

    results_df = pd.DataFrame.from_dict(results)

    ax.set_xlabel(r"Budget / min budget of any solver", fontsize=font_size)
    ax.set_ylabel(r"Proportion problems solved", fontsize=font_size)
    ax.legend(loc='lower right', fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.axis([np.min(xvals), np.max(xvals), 0, 1])  # (xlow, xhigh, ylow, yhigh)
    ax.grid()

    # Nicely format x-axis labels
    log_xmax = int(round(math.log(np.max(xvals), 2.0)))
    xticks = [2 ** y for y in range(log_xmax + 1)]  # 1, 2, 4, 8, ..., max(xvals)
    ax.set_xticks(xticks)
    ax.minorticks_off()  # in newer matploblib versions, minor ticks break label changes for log-scale axes
    # ax.set_xticks(range(1, xticks[-1] + 1), minor=True)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    if filestem is not None:
        results_df.to_csv("%s_perf%g_raw.csv" % (filestem, tau_level), index=False)
        plt.savefig("%s_perf%g.%s" % (filestem, tau_level, fmt), bbox_inches='tight')
    else:
        plt.show()

    return


def make_profiles(plot_info, solve_times, tau_levels, filestem, budget_in_gradients, plot_data=True, plot_perf=True, fmt='pdf'):
    for tau_level in tau_levels:
        if plot_data:
            data_profile = build_data_profile_curves(solve_times, budget_in_gradients=budget_in_gradients, tau_level=tau_level)
            plot_data_profile(data_profile, plot_info, tau_level, fmt=fmt, filestem=filestem)

        if plot_perf:
            perf_profile = build_perf_profile_curves(solve_times, tau_level=tau_level, max_ratio=32.0)
            plot_perf_profile(perf_profile, plot_info, tau_level, fmt=fmt, filestem=filestem)
    return


def iter_count_check(all_results):
    iter_counts = {'run_name': [], 'probkey': [], 'niters': [],
                   'n_successful_tangent': [], 'n_successful_normal': [], 'n_unsuccessful': []}
    for run_name in all_results:
        for probkey in all_results[run_name]:
            n_successful_tangent = all_results[run_name][probkey]['results']['niters_successful_tangent']
            n_successful_normal = all_results[run_name][probkey]['results']['niters_successful_normal']
            n_unsuccessful = all_results[run_name][probkey]['results']['niters_unsuccessful']
            iter_counts['run_name'].append(run_name)
            iter_counts['probkey'].append(probkey)
            iter_counts['niters'].append(n_successful_tangent + n_successful_normal + n_unsuccessful)
            iter_counts['n_successful_tangent'].append(n_successful_tangent)
            iter_counts['n_successful_normal'].append(n_successful_normal)
            iter_counts['n_unsuccessful'].append(n_unsuccessful)
    return pd.DataFrame.from_dict(iter_counts)

def main():
    # task = 'process_raw_results'  # create solve_times.csv
    # task = 'make_profiles'
    task = 'iter_count_check'

    solve_times_file = os.path.join('raw_results', 'solve_times.csv')
    tau_levels = [1, 2, 3, 4, 5, 6, 7]
    budget_in_gradients = 200

    if task == 'process_raw_results':
        probsets = []
        probsets.append('BOUNDS_ONLY')
        probsets.append('HAS_LINCONS')

        run_names = []
        run_names.append('tangent_only')
        run_names.append('tangent_only_simple_rho')
        run_names.append('tangent_and_normal')

        all_results = load_all_results(run_names, probsets, drop_missing_results=True)

        solve_times = get_solve_times(all_results, tau_levels)
        solve_times.to_csv(solve_times_file, index=False)

    elif task == 'make_profiles':
        solve_times = pd.read_csv(solve_times_file)
        print(solve_times.head())
        print(solve_times.tail())

        TANGENT_ONLY_PLOT_INFO = {'run_name': 'tangent_only', 'col': 'k', 'ls': '-', 'lbl': 'T (full dec)', 'mkr': '', 'ms': 0}
        TANGENT_ONLY_SIMPLE_PLOT_INFO = {'run_name': 'tangent_only_simple_rho', 'col': 'b', 'ls': '-', 'lbl': 'T (simple dec)', 'mkr': '', 'ms': 0}
        TANGENT_AND_NORMAL_PLOT_INFO = {'run_name': 'tangent_and_normal', 'col': 'r', 'ls': '-', 'lbl': 'T+N (simple dec)', 'mkr': '', 'ms': 0}

        FULL_PLOT_INFO = []
        FULL_PLOT_INFO.append(TANGENT_ONLY_PLOT_INFO)
        FULL_PLOT_INFO.append(TANGENT_ONLY_SIMPLE_PLOT_INFO)
        FULL_PLOT_INFO.append(TANGENT_AND_NORMAL_PLOT_INFO)

        filestem = os.path.join('profiles', 'main_comparison')
        probsets = ['BOUNDS_ONLY', 'HAS_LINCONS']
        make_profiles(FULL_PLOT_INFO, solve_times[solve_times["probset"].isin(probsets)], tau_levels, filestem, budget_in_gradients,
                      plot_data=True, plot_perf=True, fmt='pdf')

        filestem = os.path.join('profiles', 'bounds_only')
        probsets = ['BOUNDS_ONLY']
        make_profiles(FULL_PLOT_INFO, solve_times[solve_times["probset"].isin(probsets)], tau_levels, filestem,
                      budget_in_gradients, plot_data=True, plot_perf=True, fmt='pdf')

        filestem = os.path.join('profiles', 'has_lincons')
        probsets = ['HAS_LINCONS']
        make_profiles(FULL_PLOT_INFO, solve_times[solve_times["probset"].isin(probsets)], tau_levels, filestem,
                      budget_in_gradients, plot_data=True, plot_perf=True, fmt='pdf')

    elif task == 'iter_count_check':
        probsets = []
        probsets.append('BOUNDS_ONLY')
        probsets.append('HAS_LINCONS')

        run_names = []
        run_names.append('tangent_only')
        run_names.append('tangent_only_simple_rho')
        run_names.append('tangent_and_normal')

        all_results = load_all_results(run_names, probsets, drop_missing_results=True)

        iter_counts = iter_count_check(all_results)
        print(iter_counts.head())
        print(iter_counts.tail())

        # Histogram 1: fraction of iterations which are successful (compare both runs)
        iter_counts['frac_successful'] = (iter_counts['n_successful_tangent'] + iter_counts['n_successful_normal']) / (iter_counts['n_successful_tangent'] + iter_counts['n_successful_normal'] + iter_counts['n_unsuccessful'])
        frac_successful_t = iter_counts[iter_counts['run_name'] == 'tangent_only']['frac_successful'].values
        frac_successful_ts = iter_counts[iter_counts['run_name'] == 'tangent_only_simple_rho']['frac_successful'].values
        frac_successful_tn = iter_counts[iter_counts['run_name'] == 'tangent_and_normal']['frac_successful'].values

        font_size = 'large'  # x-large for presentations
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.clf()
        ax = plt.gca()
        counts, bins = np.histogram(frac_successful_t, bins=20)
        ax.plot(0.5 * (bins[1:] + bins[:-1]), counts, 'k-', label='T (full dec)')

        counts, bins = np.histogram(frac_successful_ts, bins=20)
        ax.plot(0.5 * (bins[1:] + bins[:-1]), counts, 'b-', label='T (simple dec)')

        counts, bins = np.histogram(frac_successful_tn, bins=20)
        ax.plot(0.5 * (bins[1:] + bins[:-1]), counts, 'r--', label='T+N (simple dec)')
        # plt.hist(frac_successful_t, bins=20, label='T')
        # plt.hist(frac_successful_tn, bins=20, label='T+N')
        ax.set_xlabel(r"Fraction of successful iterations", fontsize=font_size)
        ax.set_ylabel(r"Number of problems", fontsize=font_size)
        ax.legend(loc='best', fontsize=font_size)
        ax.tick_params(axis='both', which='major', labelsize=font_size)
        ax.axis([0, 1, 0, None])  # (xlow, xhigh, ylow, yhigh)
        ax.grid()
        plt.savefig(os.path.join('profiles', 'frac_successful.pdf'), bbox_inches='tight')

        # Histogram 2: for tangent_and_normal, fraction of successful iterations which came from N
        iter_counts['frac_successful_normal'] = iter_counts['n_successful_normal'] / (iter_counts['n_successful_tangent'] + iter_counts['n_successful_normal'])
        frac_successful_normal = iter_counts[iter_counts['run_name'] == 'tangent_and_normal']['frac_successful_normal'].values

        font_size = 'large'  # x-large for presentations
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.clf()
        ax = plt.gca()
        ax.hist(frac_successful_normal, color='r', bins=20)
        ax.set_xlabel(r"Fraction of successful iterations using normal direction", fontsize=font_size)
        ax.set_ylabel(r"Number of problems", fontsize=font_size)
        # ax.legend(loc='best', fontsize=font_size)
        ax.tick_params(axis='both', which='major', labelsize=font_size)
        ax.axis([0, 1, 0, None])  # (xlow, xhigh, ylow, yhigh)
        ax.grid()
        plt.savefig(os.path.join('profiles', 'frac_successful_normal.pdf'), bbox_inches='tight')

    else:
        raise RuntimeError("Unknown task '%s'" % task)
    print("Done")
    return


if __name__ == "__main__":
    main()
