"""Full-cohort, train-fitted preprocessing with independent isotonic calibration.

Prespecified 64/16/20 splits, ten seeds, identical fit partitions for all models.
Each job checkpoints its results and raw test scores. Existing jobs are reused
only when their configuration and source-file hashes match.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd
import psutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from emdt import ExactMonotonicTree
from run_revision_nmi import NonNegativeLogistic, expected_calibration_error, safe_predict_score

HERE=Path(__file__).resolve().parent
OUT=HERE/'full_cohort'
SOURCES={'compas_real':Path('D:/data/compas/compas-scores-two-years.csv'),
         'german_real':Path('D:/data/german/german.data'),
         'adult_income':Path('D:/data/adult/adult.data'),
         'bank_marketing':Path('D:/data/bank/bank-full.csv')}
CONFIG={'protocol':'full_cohort_v1','seeds':list(range(42,52)), 'test_fraction':.2,
        'validation_fraction_of_outer_train':.2, 'depth':3,'solver_limit':30,
        'solver_workers':1,'bootstrap_repeats':2000,'bootstrap_seed':123,
        'audit_pairs':2000, 'compress_rows':True,'constant_hint':True,
        'feature_set':'same numeric policy features as prior main experiment',
        'scaling':'fit minima/maxima on fit split; clip transformed evaluation data to [0,1]',
        'calibration':'isotonic map on independent validation split; reuse same fitted EMDT',
        'models':['CART','ODT','EMDT','EMDT-Cell','EMDT-Iso','MonoXGB1','MonoRF',
                  'XGB-Mono','LGBM-Mono','LGBM-Adv','MonoScore']}


def source_hashes():
    return {k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in SOURCES.items()}


def load_raw(name):
    path=SOURCES[name]
    if name=='compas_real':
        df=pd.read_csv(path)
        raw_n=len(df)
        df=df[df.days_b_screening_arrest.between(-30,30)&(df.is_recid!=-1)&
              (df.c_charge_degree!='O')&(df.score_text!='N/A')]
        features=['age','priors_count','juv_fel_count','juv_misd_count','juv_other_count']
        df=df[features+['two_year_recid']].dropna()
        y=1-df.two_year_recid.to_numpy(dtype=int)
        directions=[1,-1,-1,-1,-1]
    elif name=='german_real':
        df=pd.read_csv(path,sep=r'\s+',header=None)
        raw_n=len(df); features=[1,4,7,10,12]
        df=df[features+[20]].dropna()
        y=(df[20].to_numpy()==1).astype(int)
        directions=[-1,-1,-1,1,1]
    elif name=='adult_income':
        cols=['age','workclass','fnlwgt','education','education_num','marital_status',
              'occupation','relationship','race','sex','capital_gain','capital_loss',
              'hours_per_week','native_country','income']
        df=pd.read_csv(path,names=cols,sep=r',\s*',engine='python',na_values=['?'])
        raw_n=len(df); features=['age','education_num','capital_gain','hours_per_week']
        df=df[features+['income']].dropna()
        y=(df.income.astype(str).str.strip()=='>50K').to_numpy(dtype=int)
        directions=[1]*4
    else:
        df=pd.read_csv(path,sep=';')
        raw_n=len(df); features=['balance','duration','campaign','previous']
        df=df[features+['y']].dropna()
        y=(df.y.astype(str).str.strip()=='yes').to_numpy(dtype=int)
        directions=[1,1,-1,1]
    feature_names={'german_real':['duration','amount','installment','residence','age']}.get(name,features)
    return dict(X=df[features].to_numpy(dtype=float),y=y,source_rows=df.index.to_numpy(),
                features=feature_names,directions=directions,raw_n=raw_n)


def prepare(data, seed):
    idx=np.arange(len(data['y']))
    outer, test=train_test_split(idx,test_size=.2,random_state=seed,stratify=data['y'])
    fit,val=train_test_split(outer,test_size=.2,random_state=seed+10000,
                            stratify=data['y'][outer])
    assert not (set(fit)&set(val) or set(fit)&set(test) or set(val)&set(test))
    raw=data['X']; lo=raw[fit].min(axis=0); hi=raw[fit].max(axis=0)
    span=np.where(hi>lo,hi-lo,1.)
    scaled=(raw-lo)/span
    scaled[:,hi==lo]=0
    clip_fraction=float(np.mean((scaled[test]<0)|(scaled[test]>1)))
    scaled=np.clip(scaled,0.,1.)
    scaled[:,np.array(data['directions'])<0]=1-scaled[:,np.array(data['directions'])<0]
    return dict(Xfit=scaled[fit],Xval=scaled[val],Xtest=scaled[test],
                yfit=data['y'][fit],yval=data['y'][val],ytest=data['y'][test],
                fit=fit,val=val,test=test,lo=lo,hi=hi,clip_fraction=clip_fraction)


def models(d,seed):
    shared=dict(max_depth=3,time_limit=CONFIG['solver_limit'],num_workers=1,
                random_seed=seed,compress_rows=True,constant_hint=True)
    xgb=dict(max_depth=3,learning_rate=.05,objective='binary:logistic',
             eval_metric='logloss',monotone_constraints=tuple([1]*d),
             tree_method='hist',random_state=seed,n_jobs=1)
    lgb=dict(n_estimators=250,max_depth=3,num_leaves=8,learning_rate=.05,
             objective='binary',monotone_constraints=[1]*d,
             random_state=seed,n_jobs=1,verbosity=-1)
    return {'CART':DecisionTreeClassifier(max_depth=3,random_state=seed),
            'ODT':ExactMonotonicTree(monotonic=False,**shared),
            'EMDT':ExactMonotonicTree(monotonic=True,**shared),
            'EMDT-Cell':ExactMonotonicTree(monotonic=True,monotonic_encoding='cells',**shared),
            'MonoXGB1':XGBClassifier(**dict(xgb,n_estimators=1,learning_rate=1)),
            'MonoRF':RandomForestClassifier(n_estimators=300,max_depth=4,
                     monotonic_cst=[1]*d,random_state=seed,n_jobs=1),
            'XGB-Mono':XGBClassifier(**dict(xgb,n_estimators=250,subsample=.9,colsample_bytree=.9)),
            'LGBM-Mono':LGBMClassifier(**dict(lgb,subsample=.9,subsample_freq=1,colsample_bytree=.9,
                                           monotone_constraints_method='basic')),
            'LGBM-Adv':LGBMClassifier(**dict(lgb,subsample=1,colsample_bytree=1,
                                          monotone_constraints_method='advanced')),
            'MonoScore':NonNegativeLogistic(l2=1e-3,max_iter=500)}


def metrics(y,score):
    pred=(score>=.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    fpr=fp/(fp+tn); fnr=fn/(fn+tp)
    return dict(Accuracy=float(accuracy_score(y,pred)),Brier=float(brier_score_loss(y,score)),
                ECE=expected_calibration_error(y,score,10),FPR=float(fpr),FNR=float(fnr),
                Cost=float(5*fpr+2*fnr),AUC=float(roc_auc_score(y,score)))


def audit(score_fn,parts,seed):
    rng=np.random.default_rng(seed+1000); d=parts['Xfit'].shape[1]; q=2000
    a,b=rng.random((q,d)),rng.random((q,d))
    low,high=np.minimum(a,b),np.maximum(a,b)
    delta=np.asarray(score_fn(low))-np.asarray(score_fn(high))
    # An empirical anchor changes one feature at a time, holding the others fixed.
    erng=np.random.default_rng(seed+2000)
    anchor=parts['Xtest'][erng.integers(0,len(parts['Xtest']),q)].copy()
    upper=anchor.copy(); j=erng.integers(0,d,q)
    upper[np.arange(q),j]+=erng.random(q)*(1-anchor[np.arange(q),j])
    edelta=np.asarray(score_fn(anchor))-np.asarray(score_fn(upper))
    return dict(viol_rate=float(np.mean(delta>1e-8)),
                viol_severity_mean=float(np.maximum(delta,0).mean()),
                empirical_viol_rate=float(np.mean(edelta>1e-8)),
                empirical_viol_severity=float(np.maximum(edelta,0).mean()))


def validate_exact(model,parts):
    full_pred=model.predict(parts['Xfit'])
    assert np.sum(full_pred!=parts['yfit'])==round(model.fit_stats_['objective'])
    assert np.array_equal(model.predict(model.training_points_),
                          model.routing_predictions_[:model.n_samples])
    if model.monotonic_encoding=='cells' and model.monotonic:
        pred=model.predict(model.cell_points_).reshape(model.cell_shape_)
        for j in range(parts['Xfit'].shape[1]):
            assert np.all(np.diff(pred,axis=j)>=0)
    elif model.monotonic:
        labels=model.solution_['leaves']
        def leaves(node):
            return [node] if node in labels else leaves(2*node)+leaves(2*node+1)
        for node in model.solution_['structure']:
            assert max(labels[l] for l in leaves(2*node))<=min(labels[l] for l in leaves(2*node+1))


def job(name,seed,fingerprint):
    warnings.filterwarnings('ignore')
    directory=OUT/'jobs'/f'{name}_{seed}'
    directory.mkdir(parents=True,exist_ok=True)
    done=directory/'results.json'
    if done.exists():
        previous=json.loads(done.read_text())
        if previous['fingerprint']!=fingerprint:
            raise ValueError(f'Checkpoint configuration mismatch: {directory}')
        return previous['rows']
    data=load_raw(name); parts=prepare(data,seed); d=parts['Xfit'].shape[1]
    np.savez_compressed(directory/'split_and_scaling.npz',fit=parts['fit'],val=parts['val'],
        test=parts['test'],source_rows=data['source_rows'],lo=parts['lo'],hi=parts['hi'])
    entries=[]; scores={}; emdt=None; emdt_row=None
    basic=dict(Dataset=name,Seed=seed,N=len(data['y']),TrainN=len(parts['fit']),
               ValidationN=len(parts['val']),TestN=len(parts['test']),Features=d,
               TestClipFraction=parts['clip_fraction'])
    for method, model in models(d,seed).items():
        start=time.perf_counter(); fitted=model.fit(parts['Xfit'],parts['yfit'])
        duration=time.perf_counter()-start
        stats=getattr(model,'fit_stats_',{})
        row=dict(basic,Method=method,Time=duration,SolverStatus=stats.get('status','NA'),
            Objective=stats.get('objective',np.nan),BestBound=stats.get('best_bound',np.nan),
            RelativeGap=stats.get('relative_gap',np.nan),
            ClippedGap=max(0.,(stats['objective']-max(0.,stats['best_bound']))/max(1.,abs(stats['objective'])))
                if fitted is not False and 'objective' in stats else np.nan,
            SolverWallTime=stats.get('solver_wall_time',np.nan),
            TrainingPatterns=stats.get('n_training_patterns',np.nan),
            CandidateSplits=getattr(model,'n_splits',np.nan),
            Cells=stats.get('n_cells',0),CalibTime=0.,
            RSS_MB=psutil.Process().memory_info().rss/2**20)
        if fitted is False:
            row.update({m:np.nan for m in ['Accuracy','Brier','ECE','FPR','FNR','Cost','AUC',
                         'viol_rate','viol_severity_mean','empirical_viol_rate',
                         'empirical_viol_severity','TrainAccuracy']})
        else:
            score_fn=lambda x,m=model:safe_predict_score(m,x)
            score=score_fn(parts['Xtest']); scores[method]=score
            row.update(metrics(parts['ytest'],score),**audit(score_fn,parts,seed))
            row['TrainAccuracy']=float(np.mean((score_fn(parts['Xfit'])>=.5)==parts['yfit']))
            if isinstance(model,ExactMonotonicTree):
                validate_exact(model,parts)
                (directory/(method+'_tree.json')).write_text(json.dumps(dict(
                    structure={str(k):[int(f),float(t)] for k,(f,t) in model.solution_['structure'].items()},
                    leaves={str(k):int(v) for k,v in model.solution_['leaves'].items()},
                    features=data['features'],directions=data['directions'],stats=stats),indent=2))
            if method=='EMDT':
                emdt=model; emdt_row=row
        entries.append(row)
        # Drop large cell models before fitting ensembles.
        if method=='EMDT-Cell':
            del model
    if emdt is not None:
        start=time.perf_counter()
        calibrator=IsotonicRegression(increasing=True,y_min=0,y_max=1,out_of_bounds='clip')
        calibrator.fit(emdt.predict(parts['Xval']),parts['yval'])
        elapsed=time.perf_counter()-start
        score_fn=lambda x:calibrator.predict(emdt.predict(x))
        score=score_fn(parts['Xtest']); scores['EMDT-Iso']=score
        row=dict(emdt_row,Method='EMDT-Iso',Time=emdt_row['Time']+elapsed,CalibTime=elapsed)
        row.update(metrics(parts['ytest'],score),**audit(score_fn,parts,seed))
        row['TrainAccuracy']=float(np.mean((score_fn(parts['Xfit'])>=.5)==parts['yfit']))
        entries.append(row)
        (directory/'calibration.json').write_text(json.dumps(dict(
            input_scores=[0,1],output_scores=calibrator.predict([0,1]).tolist(),
            validation_rows=len(parts['val']),base_tree='EMDT',
            base_objective_applies_to='uncalibrated hard labels'),indent=2))
    else:
        row=dict(emdt_row or basic,Method='EMDT-Iso',SolverStatus='UNKNOWN')
        entries.append(row)
    np.savez_compressed(directory/'test_scores.npz',y=parts['ytest'],**scores)
    done.write_text(json.dumps(dict(fingerprint=fingerprint,rows=entries),indent=2))
    return entries


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--seeds',default=','.join(map(str,CONFIG['seeds'])))
    parser.add_argument('--datasets',default=','.join(SOURCES))
    args=parser.parse_args()
    seeds=[int(s) for s in args.seeds.split(',')]
    datasets=args.datasets.split(',')
    OUT.mkdir(exist_ok=True)
    hashes=source_hashes()
    fingerprint=hashlib.sha256(json.dumps(dict(config=CONFIG,source_hashes=hashes),
                                         sort_keys=True).encode()).hexdigest()
    packages=['numpy','pandas','scipy','ortools','scikit-learn','xgboost','lightgbm','psutil']
    manifest=dict(config=CONFIG,source_hashes=hashes,fingerprint=fingerprint,
        python=sys.version,executable=sys.executable,platform=platform.platform(),
        packages={p:importlib.metadata.version(p) for p in packages},parallel_jobs=args.workers,
        memory_limit_gb=28,selected_environment='project .venv with installed CPU stack',
        note='Existing project stack retained because solver environments lack OR-Tools/boosting; GPU not used.',
        source_code_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [Path(__file__),HERE/'emdt.py']})
    (OUT/'environment.json').write_text(json.dumps(manifest,indent=2))
    tasks=[(ds,seed) for ds in datasets for seed in seeds]
    records=[]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures={executor.submit(job,ds,seed,fingerprint):(ds,seed) for ds,seed in tasks}
        for future in as_completed(futures):
            rows=future.result(); records.extend(rows)
            pd.DataFrame(records).sort_values(['Dataset','Seed','Method']).to_csv(OUT/'runs.csv',index=False)
            ds,seed=futures[future]
            exact=[r for r in rows if r['Method'] in ['ODT','EMDT','EMDT-Cell']]
            print(f'{len(records)//11}/{len(tasks)} {ds} seed={seed}: '+
                  ', '.join(f"{r['Method']} {r['SolverStatus']} acc={r['Accuracy']:.3f}" for r in exact),flush=True)
            if psutil.virtual_memory().used > 28*2**30:
                warnings.warn('Total system memory exceeds 28 GiB; monitor process sizes.')
    print('Stored',len(records),'method evaluations',flush=True)


if __name__=='__main__':
    main()
