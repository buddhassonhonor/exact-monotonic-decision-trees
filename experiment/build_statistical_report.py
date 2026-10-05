"""Recompute complete uncertainty reports from the stored experimental runs.

Also evaluates the single-tree boosting baseline on the SAME 500-record cohorts.
Existing benchmark results are never overwritten.
"""
from pathlib import Path
import json
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import scipy
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import xgboost
import sklearn
from run_revision_nmi import (load_compas_dataset, load_german_dataset,
    load_adult_dataset, load_bank_dataset, evaluate_model)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NAMES = dict(compas_real='COMPAS', german_real='German', adult_income='Adult', bank_marketing='Bank')
MODEL_NAMES = {'XGB-Mono-Best':'XGB-B', 'LGBM-Mono-Best':'LGBM-B'}


def ci(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return np.nan, np.nan
    rng = np.random.default_rng(123)
    means = rng.choice(a, size=(2000, len(a)), replace=True).mean(axis=1)
    return tuple(np.quantile(means, [.025, .975]))


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p)-np.arange(len(p))) * p[order])
    out = np.empty_like(p)
    out[order] = np.minimum(adjusted, 1)
    return out


def main():
    warnings.filterwarnings('ignore')
    datasets = [fn() for fn in [load_compas_dataset, load_german_dataset,
                               load_adult_dataset, load_bank_dataset]]
    baseline_path = HERE / 'single_tree_runs.csv'
    rows, metadata = [], []
    for data in datasets:
        X, y, name = data['X'], data['y'], data['name']
        splits = []
        for seed in range(42, 52):
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=seed, stratify=y)
            splits.append(sum(4 if len(np.unique(Xtr[:, j])) > 5 else len(np.unique(Xtr[:, j]))-1
                              for j in range(X.shape[1])))
            if not baseline_path.exists():
                model = XGBClassifier(n_estimators=1, max_depth=3, learning_rate=1,
                    objective='binary:logistic', eval_metric='logloss', tree_method='hist',
                    monotone_constraints=tuple([1]*X.shape[1]), random_state=seed, n_jobs=1)
                row = evaluate_model(model, Xtr, ytr, Xte, yte, X.shape[1], seed)
                rows.append(dict(row, Dataset=name, Method='MonoXGB1', Seed=seed))
        metadata.append(dict(Dataset=name, UsedN=len(X), TrainN=len(Xtr), TestN=len(Xte),
                             Features=X.shape[1], Governed=X.shape[1], Kmin=min(splits),
                             Kmax=max(splits), FeatureNames=';'.join(data['feature_names'])))
    if rows:
        pd.DataFrame(rows).to_csv(baseline_path, index=False)
    pd.DataFrame(metadata).to_csv(HERE / 'benchmark_cohort_audit.csv', index=False)
    runs = pd.concat([pd.read_csv(HERE/'revision_runs.csv'), pd.read_csv(baseline_path)], ignore_index=True)
    assert not runs.duplicated(['Dataset','Method','Seed']).any()
    metrics = ['Accuracy','Brier','ECE','Cost','FPR','FNR','viol_rate','viol_severity_mean','Time']
    summary = []
    for (dataset, method), block in runs.groupby(['Dataset','Method']):
        assert sorted(block.Seed) == list(range(42, 52))
        r = dict(Dataset=dataset, Method=method, Runs=len(block))
        for metric in metrics:
            a = block[metric].to_numpy()
            lo, hi = ci(a)
            r.update({metric+'_Mean':a.mean(), metric+'_Std':a.std(ddof=0),
                      metric+'_CI_L':lo, metric+'_CI_U':hi})
        summary.append(r)
    summary = pd.DataFrame(summary)
    summary.to_csv(HERE/'complete_benchmark_statistics.csv', index=False)
    tests = []
    for dataset, block in runs.groupby('Dataset'):
        e = block[block.Method=='EMDT'].set_index('Seed').sort_index()
        for method in sorted(set(block.Method)-{'EMDT'}):
            b = block[block.Method==method].set_index('Seed').sort_index()
            for metric in ['Accuracy','Cost']:
                diff = e[metric].to_numpy()-b[metric].to_numpy()
                p = 1. if np.all(diff==0) else wilcoxon(diff, zero_method='pratt',
                        alternative='two-sided', method='auto').pvalue
                lo, hi = ci(diff)
                tests.append(dict(Dataset=dataset, Baseline=method, Metric=metric, N=len(diff),
                                  Difference=diff.mean(), CI_L=lo, CI_U=hi, P=p))
    tests = pd.DataFrame(tests)
    tests['P_Holm'] = holm(tests.P)
    tests.to_csv(HERE/'complete_paired_tests.csv', index=False)
    syn = pd.read_csv(HERE/'synthetic_30seeds_v2_raw.csv')
    sr = []
    for (dataset, noise), block in syn.groupby(['Dataset','Noise']):
        assert sorted(block.Seed)==list(range(42,72))
        diff = block.EMDT_acc.to_numpy()-block.CART_acc.to_numpy()
        lo, hi = ci(diff)
        r = dict(Dataset=dataset, Noise=noise, N=len(diff), Difference=diff.mean(),
                 CI_L=lo, CI_U=hi, P=wilcoxon(diff, zero_method='pratt', method='auto').pvalue,
                 Optimal=int((block.EMDT_status=='OPTIMAL').sum()),
                 Feasible=int((block.EMDT_status=='FEASIBLE').sum()))
        for method in ['CART','EMDT']:
            r[method+'_CI_L'], r[method+'_CI_U'] = ci(block[method+'_acc'])
        sr.append(r)
    sr = pd.DataFrame(sr)
    sr['P_Holm']=holm(sr.P)
    sr.to_csv(HERE/'synthetic_uncertainty.csv', index=False)
    # Neutral, publication-facing supplementary tables.
    tex = [r'\appendix', r'\section{Complete uncertainty estimates}\label{app:uncertainty}',
           'Intervals are percentile 95\\% bootstrap intervals for the mean over repeated splits, '
           'using 2,000 resamples and bootstrap seed 123. They describe split variability on a fixed '
           'cohort; overlapping training and test sets limit population-level inference. '
           'XGB-B and LGBM-B denote the stronger boosting configurations. '
           'All real-data blocks contain ten seeds (42--51).']
    for cols, title in [(['Accuracy','ECE','Brier'],'Predictive and calibration metrics'),
                        (['Cost','FPR','FNR'],'Cost and error rates'),
                        (['viol_rate','viol_severity_mean','Time'],'Sampled monotonicity audits and fit time')]:
        tex += [r'\begingroup\scriptsize\setlength{\tabcolsep}{3pt}',
                r'\begin{longtable}{@{}llccc@{}}',
                '\\caption{'+title+'; mean [95\\% interval].}\\\\',
                r'\toprule Dataset & Method & '+ ' & '.join(c.replace('_',r'\_') for c in cols)+r' \\',
                r'\midrule\endfirsthead',r'\toprule Dataset & Method & '+ ' & '.join(c.replace('_',r'\_') for c in cols)+r' \\',
                r'\midrule\endhead']
        for _, row in summary.iterrows():
            cells=[NAMES[row.Dataset],MODEL_NAMES.get(row.Method,row.Method)]
            cells += [f'{row[m+"_Mean"]:.3f} [{row[m+"_CI_L"]:.3f}, {row[m+"_CI_U"]:.3f}]' for m in cols]
            tex += [' & '.join(cells)+r' \\']
        tex += [r'\bottomrule\end{longtable}\endgroup']
    tex += [r'\section{Paired comparisons}\label{app:paired}',
        'Differences are EMDT minus the comparator; positive accuracy differences and negative cost '
        'differences favor EMDT. Two-sided Wilcoxon signed-rank tests use the Pratt treatment of zeros; '
        'identical paired outcomes have $p=1$. Holm correction is applied jointly to all 80 real-data '
        'tests (four datasets, ten comparators, two metrics). These are exploratory comparisons '
        'across overlapping repeated splits.']
    for dataset, block in tests.groupby('Dataset'):
        tex += [r'\begin{table}[ht]\centering\scriptsize\setlength{\tabcolsep}{4pt}',
            '\\caption{'+NAMES[dataset]+': paired mean differences and raw/adjusted $p$-values.}',
            r'\begin{tabular}{@{}lrrrrrr@{}}\toprule',
            r'Method & $\Delta$Acc & $p$ & $p_{\rm Holm}$ & $\Delta$Cost & $p$ & $p_{\rm Holm}$ \\\midrule']
        for method, grp in block.groupby('Baseline'):
            a=grp[grp.Metric=='Accuracy'].iloc[0]; c=grp[grp.Metric=='Cost'].iloc[0]
            tex += [f'{MODEL_NAMES.get(method,method)} & {a.Difference:.3f} & {a.P:.4f} & {a.P_Holm:.4f} & {c.Difference:.3f} & {c.P:.4f} & {c.P_Holm:.4f}'+r' \\']
        tex += [r'\bottomrule\end{tabular}\end{table}', r'\FloatBarrier']
    tex += [r'\begin{table}[ht]\centering\scriptsize',
        r'\caption{Synthetic accuracy differences (EMDT minus CART), paired 95\% intervals, tests, and EMDT solver status counts (30 seeds).}\label{tab:syntheticunc}',
        r'\begin{tabular}{@{}lcrrrcc@{}}\toprule',
        r'Task & Noise & $\Delta$Acc [CI] & $p$ & $p_{\rm Holm}$ & Opt. & Feas. \\\midrule']
    for _,r in sr.iterrows():
        tex += [f'{r.Dataset} & {r.Noise:.2f} & {r.Difference:.3f} [{r.CI_L:.3f}, {r.CI_U:.3f}] & {r.P:.4f} & {r.P_Holm:.4f} & {r.Optimal} & {r.Feasible}'+r' \\']
    tex += [r'\bottomrule\end{tabular}\end{table}', r'\FloatBarrier',
        'Holm correction for synthetic accuracy uses a separate family of four tests.']
    (ROOT/'statistical_appendix.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')
    (HERE/'statistical_report_environment.json').write_text(json.dumps(dict(
        python=sys.version, platform=platform.platform(), numpy=np.__version__, pandas=pd.__version__,
        scipy=scipy.__version__, sklearn=sklearn.__version__, xgboost=xgboost.__version__,
        bootstrap_repeats=2000, bootstrap_seed=123, cohort_seed=2026,
        note='Only MonoXGB1 was refitted; all other uncertainty estimates use archived runs.'),indent=2),encoding='utf-8')
    print(pd.DataFrame(metadata).to_string(index=False))
    print(sr.to_string(index=False))
    print('Holm-significant real-data tests:', int((tests.P_Holm < .05).sum()))
    print(summary[summary.Method=='MonoXGB1'][['Dataset','Accuracy_Mean','Accuracy_Std','ECE_Mean','Cost_Mean','Time_Mean']].to_string(index=False))


if __name__=='__main__':
    main()
