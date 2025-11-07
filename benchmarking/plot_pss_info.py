"""
Plot Lambda-PSS summary information
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from make_profiles import load_all_results_single_run


def summarize_lambda_info():
    # Solve for v using DIRECT
    run_name = 'tangent_and_normal_detailed'
    probset = 'HAS_LINCONS'
    all_results = load_all_results_single_run(run_name, [probset])

    summary = {'probname': [], 'n': [], 'generator': [], 'lambda': [], 'lambda_crit': []}

    for prob in all_results:
        prob_results = all_results[prob]
        nvals = len(prob_results['results']['detailed_info']['generator'])
        summary['probname'] += [prob_results['name']] * nvals
        summary['n'] += [prob_results['n']] * nvals
        summary['generator'] += prob_results['results']['detailed_info']['generator']
        summary['lambda'] += prob_results['results']['detailed_info']['lambda_alpha']
        summary['lambda_crit'] += prob_results['results']['detailed_info']['lambda_alpha_gradf']

    df = pd.DataFrame.from_dict(summary)
    df['lambda_norm'] = df['lambda'] / np.sqrt(df['n'])
    df['lambda_crit_norm'] = df['lambda_crit'] / np.sqrt(df['n'])
    return df

def plot_info(df, whole_region=True):
    # whole_region = True --> max Lambda for whole feasible regions
    # whole_region = False --> Lambda for v corresponding to (true) criticality measure only
    key = 'lambda_norm' if whole_region else 'lambda_crit_norm'

    if whole_region:
        df = df[(df['lambda'] > 0.0)].dropna()  # remove lambda=0 and rows with NaN values
    else:
        df = df[np.isfinite(df['lambda_crit'])].dropna()  # remove lambda=0 and rows with NaN values

    # Rename/aggregate generator types
    gen_map = {'unconstrained': 'unconstrained',
               'pseudoinverse': 'linearly_independent',
               'double_descent': 'double_description',
               'double_descent_recursive': 'recursive',
               'double_descent_recursive_normal': 'recursive',
               'normal_cone': 'normal_generators'}
    df['generator'] = df['generator'].map(gen_map)

    generators = sorted(list(df['generator'].unique()))
    gen_count = {}
    for g in generators:
        gen_count[g] = len(df[df['generator'] == g])
    print(gen_count)

    plt.figure(figsize=(8,5))
    font_size = 'large'  # x-large for presentations
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')



    generators = ['unconstrained', 'linearly_independent', 'double_description', 'normal_generators', 'recursive']
    gen_lbls = ['Unconstrained\n$N=%g$' % gen_count['unconstrained'],
                'Full rank\n$N=%g$' % gen_count['linearly_independent'],
                'Double desc.\n$N=%g$' % gen_count['double_description'],
                'Normal gens.\n$N=%g$' % gen_count['normal_generators'],
                'Recursive\n$N=%g$' % gen_count['recursive']]

    # generators = ['unconstrained', 'pseudoinverse', 'double_descent', 'double_descent_recursive', 'double_descent_recursive_normal', 'normal_cone']
    # gen_lbls = ['Unconstrained\n$N=%g$' % gen_count['unconstrained'],
    #             'Full rank\n$N=%g$' % gen_count['pseudoinverse'],
    #             'Double descent\n$N=%g$' % gen_count['double_descent'],
    #             'DD+R\n$N=%g$' % gen_count['double_descent_recursive'],
    #             'DD+R + empty tangent\n$N=%g$' % gen_count['double_descent_recursive_normal'],
    #             'Empty tangent\n$N=%g$' % gen_count['normal_cone']]

    # plt.violinplot([df[df['generator'] == g]['lambda_norm'].values for g in generators], showmedians=True)
    plt.boxplot([df[df['generator'] == g][key].values for g in generators])
    plt.gca().set_xticks([n+1 for n in range(len(generators))], gen_lbls)
    if whole_region or True:
        plt.gca().set_yscale('log')
    plt.gca().yaxis.grid(True)
    plt.gca().tick_params(axis='both', which='major', labelsize=font_size)
    plt.xlabel('Poll set type', fontsize=font_size)
    if whole_region:
        plt.ylabel(r"PSS quality, $\Lambda/\sqrt{n}$", fontsize=font_size)
        plt.savefig('lambda_info.pdf', bbox_inches='tight')
    else:
        plt.ylabel(r"PSS quality at criticality point, $\Lambda/\sqrt{n}$", fontsize=font_size)
        plt.savefig('lambda_info_crit.pdf', bbox_inches='tight')
    return

def main():
    # Compare estimates of Lambda-PSS for linearly constrained problems (not bound constraints)
    df = summarize_lambda_info()
    df.to_csv('lambda_info.csv', index=False)
    # print(df.head())
    # print(df.tail())
    # df.to_csv(os.path.join('raw_results', 'lambda_pss_summary.csv'), index=False)
    plot_info(df, whole_region=True)  # lambda_info.pdf
    # plot_info(df, whole_region=False)  # lambda_info_crit.pdf
    print("Done")
    return

if __name__ == '__main__':
    main()
