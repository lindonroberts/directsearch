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

    summary = {'probname': [], 'n': [], 'generator': [], 'lambda': []}

    for prob in all_results:
        prob_results = all_results[prob]
        nvals = len(prob_results['results']['detailed_info']['generator'])
        summary['probname'] += [prob_results['name']] * nvals
        summary['n'] += [prob_results['n']] * nvals
        summary['generator'] += prob_results['results']['detailed_info']['generator']
        summary['lambda'] += prob_results['results']['detailed_info']['lambda_alpha']

    df = pd.DataFrame.from_dict(summary)
    df['lambda_norm'] = df['lambda'] / np.sqrt(df['n'])
    return df

def plot_info(df):
    df = df[df['lambda'] > 0.0].dropna()  # remove lambda=0 and rows with NaN values
    generators = sorted(list(df['generator'].unique()))
    gen_count = {}
    for g in generators:
        gen_count[g] = len(df[df['generator'] == g])
    print(gen_count)

    plt.figure(figsize=(8,5))
    font_size = 'large'  # x-large for presentations
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    generators = ['unconstrained', 'pseudoinverse', 'double_descent', 'normal_cone']
    gen_lbls = ['Unconstrained\n$N=%g$' % gen_count['unconstrained'],
                'Full rank\n$N=%g$' % gen_count['pseudoinverse'],
                'Double descent\n$N=%g$' % gen_count['double_descent'],
                'Empty tangent\n$N=%g$' % gen_count['normal_cone']]

    # plt.violinplot([df[df['generator'] == g]['lambda_norm'].values for g in generators], showmedians=True)
    plt.boxplot([df[df['generator'] == g]['lambda_norm'].values for g in generators])
    plt.gca().set_xticks([n+1 for n in range(len(generators))], gen_lbls)
    plt.gca().set_yscale('log')
    plt.gca().yaxis.grid(True)
    plt.gca().tick_params(axis='both', which='major', labelsize=font_size)
    plt.xlabel('Poll set type', fontsize=font_size)
    plt.ylabel(r"PSS quality, $\Lambda/\sqrt{n}$", fontsize=font_size)
    plt.savefig('lambda_info.pdf', bbox_inches='tight')
    return

def main():
    # Compare estimates of Lambda-PSS for linearly constrained problems (not bound constraints)
    df = summarize_lambda_info()
    # print(df.head())
    # print(df.tail())
    # df.to_csv(os.path.join('raw_results', 'lambda_pss_summary.csv'), index=False)
    plot_info(df)
    print("Done")
    return

if __name__ == '__main__':
    main()
