"""Validate manuscript main-table numbers and archive analysis provenance."""
from pathlib import Path
import hashlib
import json
import re
import pandas as pd
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def main():
    text=(ROOT/'main.tex').read_text(encoding='utf-8')
    stats=pd.read_csv(HERE/'complete_benchmark_statistics.csv')
    mapping={'COMPAS':'compas_real','German Credit':'german_real',
             'Adult Income':'adult_income','Bank Marketing':'bank_marketing'}
    mismatches=[]
    checked=0
    for line in text.splitlines():
        cells=line.split(' & ')
        if len(cells)!=8 or cells[0] not in mapping:
            continue
        method=re.sub(r'\$.*?\$','',cells[1])
        rows=stats[(stats.Dataset==mapping[cells[0]])&(stats.Method==method)]
        if rows.empty:
            continue
        row=rows.iloc[0]
        numbers=[float(x) for cell in cells[2:] for x in re.findall(r'\d+\.\d+',cell)]
        expected=[row.Accuracy_Mean,row.Accuracy_Std,row.ECE_Mean,row.Cost_Mean,
                  row.viol_rate_Mean,row.viol_severity_mean_Mean,row.Time_Mean]
        assert len(numbers)==len(expected),line
        for observed,wanted in zip(numbers,expected):
            if abs(observed-wanted) > .00050001:
                mismatches.append((cells[0],method,observed,wanted))
        checked+=1
    print('Real-data main-table rows checked:',checked)
    print('Mismatches:',mismatches)
    assert not mismatches
    runs=pd.read_csv(HERE/'revision_runs.csv')
    solver=[]
    for (d,m),grp in runs[runs.Method.isin(['ODT','EMDT'])].groupby(['Dataset','Method']):
        clipped=(grp.Objective-np.maximum(grp.BestBound,0))/np.maximum(grp.Objective.abs(),1)
        solver.append(dict(Dataset=d,Method=m,Optimal=int((grp.SolverStatus=='OPTIMAL').sum()),
                           Feasible=int((grp.SolverStatus=='FEASIBLE').sum()),
                           RawGap=grp.RelativeGap.mean(),ClippedGap=clipped.mean()))
    solver=pd.DataFrame(solver)
    solver.to_csv(HERE/'solver_bound_audit.csv',index=False)
    print(solver.to_string(index=False))
    sources=['revision_runs.csv','synthetic_30seeds_v2_raw.csv','single_tree_runs.csv',
             'model_class_fits.csv','scalability_5reps.csv','depth_ablation_10seeds.csv']
    (HERE/'analysis_input_hashes.json').write_text(json.dumps({name:hashlib.sha256(
        (HERE/name).read_bytes()).hexdigest() for name in sources},indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
