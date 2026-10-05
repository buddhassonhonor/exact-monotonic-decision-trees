"""Generate every manuscript table from immutable per-run records."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; OUT=HERE/'full_cohort'
DATASETS=['compas_real','german_real','adult_income','bank_marketing']
NAMES=dict(zip(DATASETS,['COMPAS','German','Adult','Bank']))
METHODS=['CART','ODT','EMDT','EMDT-DP','EMDT-Cell','EMDT-Iso','MonoXGB1','MonoRF',
         'XGB-Mono','LGBM-Mono','LGBM-Adv','MonoScore']
METRICS=['Accuracy','TrainAccuracy','Brier','ECE','AUC','FPR','FNR','Cost',
         'viol_rate','viol_severity_mean','empirical_viol_rate','empirical_viol_severity','Time']
LABELS=dict(TrainAccuracy='Train acc.',Accuracy='Test acc.',Brier='Brier',ECE='ECE',AUC='AUC',
            FPR='FPR',FNR='FNR',Cost='Cost',viol_rate='Uniform rate',
            viol_severity_mean='Uniform severity',empirical_viol_rate='Anchor rate',
            empirical_viol_severity='Anchor severity',Time='Fit seconds')

def ci(a):
    a=np.asarray(a,float); assert np.isfinite(a).all()
    means=np.random.default_rng(123).choice(a,(2000,len(a)),replace=True).mean(axis=1)
    return np.quantile(means,[.025,.975])

def holm(p):
    p=np.asarray(p); order=np.argsort(p); result=np.empty(len(p))
    result[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return result

def test(diff):
    return 1. if np.all(diff==0) else float(wilcoxon(diff,zero_method='pratt',method='auto').pvalue)

def longtable(caption,label,columns,header,rows):
    return '\n'.join([r'\begingroup\small\setlength{\tabcolsep}{3pt}',
        r'\begin{longtable}{@{}'+columns+r'@{}}',
        r'\caption{'+caption+r'}\label{'+label+r'}\\',
        r'\toprule '+header+r'\\\midrule\endfirsthead',
        r'\toprule '+header+r'\\\midrule\endhead',
        r'\midrule\multicolumn{'+str(len(columns))+r'}{r}{Continued on next page}\\\endfoot',
        r'\bottomrule\endlastfoot',*rows,r'\end{longtable}\endgroup'])

def main():
    runs=pd.read_csv(OUT/'all_runs.csv')
    assert len(runs)==480 and not runs.duplicated(['Dataset','Method','Seed']).any()
    records=[]
    for (dataset,method),block in runs.groupby(['Dataset','Method']):
        assert sorted(block.Seed)==list(range(42,52))
        for metric in METRICS:
            a=block[metric].to_numpy(); lo,hi=ci(a)
            records.append(dict(Dataset=dataset,Method=method,Metric=metric,Mean=a.mean(),
                SD=a.std(ddof=1),CI_L=lo,CI_U=hi,N=len(a)))
    summary=pd.DataFrame(records); summary.to_csv(OUT/'summary.csv',index=False)
    indexed=summary.set_index(['Dataset','Method','Metric'])
    tests=[]; classes=[]
    for ds in DATASETS:
        block=runs[runs.Dataset==ds]; e=block[block.Method=='EMDT'].set_index('Seed').sort_index()
        for method in METHODS:
            if method=='EMDT': continue
            b=block[block.Method==method].set_index('Seed').sort_index()
            for metric in ['Accuracy','Cost']:
                diff=e[metric].to_numpy()-b[metric].to_numpy(); lo,hi=ci(diff)
                tests.append(dict(Dataset=ds,Baseline=method,Metric=metric,Difference=diff.mean(),
                                  CI_L=lo,CI_U=hi,P=test(diff)))
        a=block[block.Method=='EMDT-DP'].set_index('Seed').sort_index()
        b=block[block.Method=='EMDT-Cell'].set_index('Seed').sort_index()
        diff=b.Accuracy.to_numpy()-a.Accuracy.to_numpy(); lo,hi=ci(diff)
        classes.append(dict(Dataset=ds,Difference=diff.mean(),CI_L=lo,CI_U=hi,P=test(diff),
            TrainErrorSaving=(a.Objective-b.Objective).mean(),
            CellCertified=int((b.SolverStatus=='OPTIMAL').sum())))
    tests=pd.DataFrame(tests); tests['HolmP']=holm(tests.P); tests.to_csv(OUT/'paired_tests.csv',index=False)
    classes=pd.DataFrame(classes); classes['HolmP']=holm(classes.P)
    classes.to_csv(OUT/'encoding_comparison.csv',index=False)
    solver=[]
    for ds in DATASETS:
        for method in ['ODT','EMDT','EMDT-Cell','EMDT-DP']:
            b=runs[(runs.Dataset==ds)&(runs.Method==method)]
            solver.append(dict(Dataset=ds,Method=method,Optimal=int(b.SolverStatus.isin(['OPTIMAL','DP-OPTIMAL']).sum()),
                Feasible=int((b.SolverStatus=='FEASIBLE').sum()),MeanObjective=b.Objective.mean(),
                MeanGap=b.ClippedGap.mean(),MeanTime=b.Time.mean(),
                Patterns=b.TrainingPatterns.mean(),Candidates=b.CandidateSplits.mean(),Cells=b.Cells.mean()))
    solver=pd.DataFrame(solver); solver.to_csv(OUT/'solver_summary.csv',index=False)
    tables=[]
    for ds in DATASETS:
        rows=[]
        for method in METHODS:
            vals=[indexed.loc[ds,method,m] for m in ['Accuracy','Cost','ECE','Time']]
            rows.append(method+' & '+ ' & '.join(f'{v.Mean:.4f} $\\pm$ {v.SD:.4f}' for v in vals)+r' \\')
        tables.append(longtable(NAMES[ds]+r': full-cohort test results, ten seeds. Mean $\pm$ sample SD; cost is $5\mathrm{FPR}+2\mathrm{FNR}$. Fit time includes construction, and EMDT-Iso reuses EMDT before validation calibration.',
            'tab:full'+ds,'lrrrr','Method & Accuracy & Cost & ECE & Seconds',rows))
    (ROOT/'benchmark_tables.tex').write_text('\n\n'.join(tables),encoding='utf-8')
    rows=[f'{NAMES[r.Dataset]} & {r.Method} & {r.Optimal}/10 & {r.Feasible}/10 & {r.MeanGap:.3f} & {r.MeanTime:.3f}'+r' \\' for r in solver.itertuples()]
    (ROOT/'solver_table.tex').write_text(longtable('Optimization evidence for the candidate-tree classes. CP-SAT uses a 30-second search budget; DP evaluates the exact cascade recurrence for the fully governed structural class. Gap uses the nonnegative clipped lower bound.','tab:solver','llrrrr','Dataset & Method & Optimal & Feasible & Gap & Seconds',rows),encoding='utf-8')
    rows=[f'{NAMES[r.Dataset]} & {r.TrainErrorSaving:.1f} & {r.Difference:.4f} & [{r.CI_L:.4f}, {r.CI_U:.4f}] & {r.HolmP:.4f}'+r' \\' for r in classes.itertuples()]
    (ROOT/'encoding_table.tex').write_text(longtable('Cell minus DP test accuracy and DP minus cell training-error count. German Cell entries use feasible-incumbent objectives. Four two-sided paired tests use Holm adjustment.','tab:encoding','lrrrr',r'Dataset & Errors saved & Acc. diff. & 95\% CI & Holm $p$',rows),encoding='utf-8')
    appendix=[r'\appendix',r'\numberwithin{table}{section}',r'\renewcommand{\thetable}{\Alph{section}.\arabic{table}}',r'\section{Complete Repeated-Split Statistics}\label{app:statistics}',
        'The tables report per-method statistics across split seeds 42--51. Standard deviations use $n-1$ in the denominator. Percentile intervals use 2,000 bootstrap resamples of the ten split-level observations, with generator seed 123.']
    for ds in DATASETS:
        rows=[]
        for method in METHODS:
            for i,metric in enumerate(METRICS):
                r=indexed.loc[ds,method,metric]
                rows.append(method+f' & {LABELS[metric]} & {r.Mean:.5f} & {r.SD:.5f} & [{r.CI_L:.5f}, {r.CI_U:.5f}]'+r' \\')
        appendix.append(longtable(NAMES[ds]+': complete statistics, ten split seeds 42--51.','tab:stats'+ds,'llrrr',r'Method & Metric & Mean & SD & 95\% CI',rows))
    appendix += [r'\section{Complete Exploratory Paired Comparisons}',
        'The difference is EMDT minus the named comparator. Positive accuracy differences favor EMDT; negative cost differences favor EMDT. Two-sided Wilcoxon signed-rank tests use the Pratt zero convention; all-zero differences have $p=1$. Holm adjustment covers all 88 accuracy/cost comparisons jointly.']
    rows=[f'{NAMES[r.Dataset]} & {r.Baseline} & {r.Metric} & {r.Difference:.4f} & [{r.CI_L:.4f}, {r.CI_U:.4f}] & {r.P:.4f} & {r.HolmP:.4f}'+r' \\' for r in tests.itertuples()]
    appendix.append(longtable('All 88 paired comparisons, including paired difference intervals.','tab:paired','lllrrrr',r'Dataset & Comparator & Metric & Difference & 95\% CI & Raw $p$ & Holm $p$',rows))
    stress=pd.read_csv(OUT/'stress_runs.csv'); synthetic=[]; syn_tests=[]
    noise=stress[stress.Study=='Noise']
    for (kind,eta),b in noise.groupby(['Generator','Noise']):
        a=b[b.Method=='EMDT-DP'].set_index('Seed').sort_index(); c=b[b.Method=='CART'].set_index('Seed').sort_index()
        diff=a.Accuracy-c.Accuracy; lo,hi=ci(diff)
        syn_tests.append(dict(Generator=kind,Noise=eta,Difference=diff.mean(),CI_L=lo,CI_U=hi,P=test(diff)))
        for method in ['CART','EMDT-DP']:
            v=b[b.Method==method]
            record=dict(Generator=kind,Noise=eta,Method=method)
            for metric in ['Accuracy','Time','viol_rate','empirical_viol_rate']:
                lo,hi=ci(v[metric]); record.update({metric:v[metric].mean(),metric+'_SD':v[metric].std(ddof=1),
                                                    metric+'_L':lo,metric+'_U':hi})
            synthetic.append(record)
    synthetic=pd.DataFrame(synthetic); synthetic.to_csv(OUT/'synthetic_summary.csv',index=False)
    syn_tests=pd.DataFrame(syn_tests); syn_tests['HolmP']=holm(syn_tests.P); syn_tests.to_csv(OUT/'synthetic_tests.csv',index=False)
    rows=[f'{r.Generator} & {r.Noise:.2f} & {r.Method} & {r.Accuracy:.3f} $\\pm$ {r.Accuracy_SD:.3f} & {r.viol_rate:.4f} & {r.Time:.4f}'+r' \\' for r in synthetic.itertuples()]
    (ROOT/'synthetic_table.tex').write_text(longtable('Synthetic results: 30 seeds per condition, $N=200$, three features, depth three. Rates use 2,000 uniform ordered pairs per seed; DP certifies all 120 fits.','tab:synthetic','lrlrrr','Generator & Flip prob. & Method & Accuracy & Viol. rate & Seconds',rows),encoding='utf-8')
    rows=[f'{r.Generator} & {r.Noise:.2f} & {r.Difference:.4f} & [{r.CI_L:.4f}, {r.CI_U:.4f}] & {r.P:.4f} & {r.HolmP:.4f}'+r' \\' for r in syn_tests.itertuples()]
    appendix.append(longtable('DP minus CART synthetic accuracy: 30 seeds, paired bootstrap intervals and four Holm-adjusted tests.','tab:syntheticunc','lrrrrr',r'Generator & Flip prob. & Difference & 95\% CI & Raw $p$ & Holm $p$',rows))
    rows=[]
    for r in synthetic.to_dict('records'):
        for metric,label in [('Accuracy','Test acc.'),('Time','Fit seconds'),('viol_rate','Uniform rate'),('empirical_viol_rate','Anchor rate')]:
            rows.append(f"{r['Generator']} & {r['Noise']:.2f} & {r['Method']} & {label} & {r[metric]:.5f} & [{r[metric+'_L']:.5f}, {r[metric+'_U']:.5f}]"+r' \\')
    appendix.append(longtable('All synthetic method-level accuracy, fit-time, and diagnostic-rate intervals, 30 seeds.','tab:synthmetrics','lrllrr',r'Generator & Flip prob. & Method & Metric & Mean & 95\% CI',rows))
    (ROOT/'full_statistical_appendix.tex').write_text('\n\n'.join(appendix),encoding='utf-8')
    print(classes.to_string(index=False)); print(syn_tests.to_string(index=False))
    print('Generated',len(summary),'metric blocks and',len(tests),'paired comparisons.')

if __name__=='__main__': main()
