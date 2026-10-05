"""Fresh noise and size/depth/feature studies; no plotting during fitting."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json
import time
import hashlib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from cascade_dp import CascadeMonotonicTree
from run_full_cohort import audit,metrics

OUT=Path(__file__).resolve().parent/'full_cohort'

def data(kind,n,d,noise,seed):
    rng=np.random.default_rng(seed)
    X=rng.random((n,d))
    probability=expit(3*(X.sum(axis=1)-d/2)) if kind=='Linear' else expit(5*(X[:,0]*X[:,1]-.25))
    y=(rng.random(n)<probability).astype(int)
    flips=rng.random(n)<noise; y[flips]=1-y[flips]
    a,b=train_test_split(np.arange(n),test_size=.2,random_state=seed,stratify=y)
    return dict(Xfit=X[a],yfit=y[a],Xtest=X[b],ytest=y[b])

def fit_job(args):
    study,kind,n,d,depth,noise,seed=args
    parts=data(kind,n,d,noise,seed); records=[]
    for method,model in [('CART',DecisionTreeClassifier(max_depth=depth,random_state=seed)),
                         ('EMDT-DP',CascadeMonotonicTree(max_depth=depth))]:
        started=time.perf_counter(); model.fit(parts['Xfit'],parts['yfit']); elapsed=time.perf_counter()-started
        score=lambda x:model.predict(x).astype(float)
        row=dict(Study=study,Generator=kind,N=n,Features=d,Depth=depth,Noise=noise,Seed=seed,
                 Method=method,Time=elapsed,TrainAccuracy=float(np.mean(model.predict(parts['Xfit'])==parts['yfit'])))
        row.update(metrics(parts['ytest'],score(parts['Xtest'])),**audit(score,parts,seed))
        if method=='EMDT-DP':
            row.update(model.fit_stats_)
            assert np.sum(model.predict(parts['Xfit'])!=parts['yfit'])==row['objective']
        records.append(row)
    return records

def main():
    jobs=[('Noise',kind,200,3,3,noise,seed) for kind in ['Linear','Polynomial']
          for noise in [.1,.25] for seed in range(42,72)]
    jobs += [('Size','Linear',n,3,3,.1,s) for n in [100,1000,10000] for s in range(42,47)]
    jobs += [('Depth','Linear',1000,3,d,.1,s) for d in [1,2,3,4] for s in range(42,47)]
    jobs += [('Features','Linear',1000,d,3,.1,s) for d in [2,3,5] for s in range(42,47)]
    with ProcessPoolExecutor(max_workers=4) as executor:
        rows=[r for batch in executor.map(fit_job,jobs) for r in batch]
    pd.DataFrame(rows).to_csv(OUT/'stress_runs.csv',index=False)
    (OUT/'stress_protocol.json').write_text(json.dumps(dict(
        generator='default_rng(seed): X, Bernoulli uniforms, flip uniforms, in that order',
        noise_seeds=list(range(42,72)),scaling='synthetic inputs already in [0,1]',
        split='stratified 80/20 with random_state=seed',audits='same two 2000-pair diagnostics as full cohort',
        hardware='same CPU stack as environment.json; 4 independent jobs',
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    print(pd.DataFrame(rows).groupby(['Study','Generator','Noise','Method']).Accuracy.mean().to_string())

if __name__=='__main__': main()
