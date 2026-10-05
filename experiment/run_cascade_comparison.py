"""Independent exact subtree-class reference on the full-cohort saved splits."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
from cascade_dp import CascadeMonotonicTree
from run_full_cohort import OUT,CONFIG,SOURCES,load_raw,prepare,metrics,audit


def run(name,seed):
    data=load_raw(name); parts=prepare(data,seed)
    directory=OUT/'jobs'/f'{name}_{seed}'
    previous=np.load(directory/'split_and_scaling.npz')
    for key in ['fit','val','test','lo','hi']:
        assert np.array_equal(previous[key],parts[key])
    model=CascadeMonotonicTree(max_depth=3)
    start=time.perf_counter(); model.fit(parts['Xfit'],parts['yfit']); elapsed=time.perf_counter()-start
    score_fn=lambda x:model.predict(x).astype(float)
    scores=score_fn(parts['Xtest']); stats=model.fit_stats_
    assert np.sum(model.predict(parts['Xfit'])!=parts['yfit'])==stats['objective']
    original=json.loads((directory/'results.json').read_text())['rows']
    for row in original:
        if row['Method']=='EMDT':
            assert row['Objective']>=stats['objective']
            if row['SolverStatus']=='OPTIMAL':
                assert row['Objective']==stats['objective']
        if row['Method']=='EMDT-Cell' and row['SolverStatus']=='OPTIMAL':
            assert row['Objective']<=stats['objective']
    row=dict(Dataset=name,Seed=seed,Method='EMDT-DP',N=len(data['y']),TrainN=len(parts['fit']),
        ValidationN=len(parts['val']),TestN=len(parts['test']),Features=parts['Xfit'].shape[1],
        TestClipFraction=parts['clip_fraction'],Time=elapsed,CalibTime=0.,
        SolverStatus='DP-OPTIMAL',Objective=stats['objective'],BestBound=stats['objective'],
        RelativeGap=0.,ClippedGap=0.,SolverWallTime=np.nan,
        TrainingPatterns=stats['n_training_patterns'],CandidateSplits=model.n_splits,Cells=0,
        States=stats['states'],TrainAccuracy=float(np.mean(model.predict(parts['Xfit'])==parts['yfit'])))
    row.update(metrics(parts['ytest'],scores),**audit(score_fn,parts,seed))
    np.savez_compressed(directory/'EMDT-DP_scores.npz',scores=scores,y=parts['ytest'])
    (directory/'EMDT-DP_tree.json').write_text(json.dumps(dict(
        structure={str(k):[int(f),float(t)] for k,(f,t) in model.solution_['structure'].items()},
        leaves={str(k):int(v) for k,v in model.solution_['leaves'].items()},
        features=data['features'],directions=data['directions'],stats=stats),indent=2))
    return row


def main():
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures=[executor.submit(run,n,s) for n in SOURCES for s in CONFIG['seeds']]
        for future in as_completed(futures):
            rows.append(future.result())
    frame=pd.DataFrame(rows).sort_values(['Dataset','Seed'])
    frame.to_csv(OUT/'cascade_runs.csv',index=False)
    merged=pd.concat([pd.read_csv(OUT/'runs.csv'),frame],ignore_index=True)
    merged.to_csv(OUT/'all_runs.csv',index=False)
    print(frame.groupby('Dataset')[['Objective','Time','Accuracy','States']].mean().to_string())
    (OUT/'cascade_code_hashes.json').write_text(json.dumps({n:hashlib.sha256(
        (Path(__file__).resolve().parent/n).read_bytes()).hexdigest()
        for n in ['cascade_dp.py','run_cascade_comparison.py']},indent=2))


if __name__=='__main__':
    main()
