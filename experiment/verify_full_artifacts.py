"""Independent source/split/prediction/summary audit; never retrains models."""
from pathlib import Path
from itertools import product
import hashlib
import json
import numpy as np
import pandas as pd
from run_full_cohort import OUT,SOURCES,load_raw,prepare,metrics,audit
from report_full_cohort import METHODS,METRICS,ci

def predict(tree,X):
    out=np.zeros(len(X),int)
    def walk(node,indices):
        if str(node) in tree['leaves']:
            out[indices]=tree['leaves'][str(node)]; return
        f,t=tree['structure'][str(node)]; right=X[indices,f]>t
        walk(2*node,indices[~right]); walk(2*node+1,indices[right])
    walk(1,np.arange(len(X)))
    return out

def verify_order(tree):
    def labels(node):
        if str(node) in tree['leaves']: return [tree['leaves'][str(node)]]
        a,b=labels(2*node),labels(2*node+1)
        assert max(a)<=min(b); return a+b
    labels(1)

def main():
    runs=pd.read_csv(OUT/'all_runs.csv'); summary=pd.read_csv(OUT/'summary.csv')
    assert len(runs)==480 and len(summary)==624
    environment=json.loads((OUT/'environment.json').read_text())
    for filename,digest in environment['source_code_hashes'].items():
        assert hashlib.sha256((OUT.parent/filename).read_bytes()).hexdigest()==digest
    for filename,digest in json.loads((OUT/'cascade_code_hashes.json').read_text()).items():
        assert hashlib.sha256((OUT.parent/filename).read_bytes()).hexdigest()==digest
    for name,path in SOURCES.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest()==environment['source_hashes'][name]
    metric_count=tree_count=0
    for name in SOURCES:
        data=load_raw(name)
        for seed in range(42,52):
            parts=prepare(data,seed); directory=OUT/'jobs'/f'{name}_{seed}'
            saved=np.load(directory/'split_and_scaling.npz')
            for key in ['fit','val','test','lo','hi']:
                assert np.array_equal(saved[key],parts[key])
            assert np.array_equal(saved['source_rows'],data['source_rows'])
            scores=np.load(directory/'test_scores.npz')
            assert np.array_equal(scores['y'],parts['ytest'])
            dp=np.load(directory/'EMDT-DP_scores.npz')
            assert np.array_equal(dp['y'],parts['ytest'])
            block=runs[(runs.Dataset==name)&(runs.Seed==seed)].set_index('Method')
            assert set(block.index)==set(METHODS)
            for method in METHODS:
                score=dp['scores'] if method=='EMDT-DP' else scores[method]
                for metric,value in metrics(parts['ytest'],score).items():
                    assert np.isclose(value,block.loc[method,metric],atol=1e-12)
                    metric_count+=1
            for method in ['ODT','EMDT','EMDT-Cell','EMDT-DP']:
                tree=json.loads((directory/(method+'_tree.json')).read_text())
                train=predict(tree,parts['Xfit']); test_scores=predict(tree,parts['Xtest'])
                assert np.sum(train!=parts['yfit'])==block.loc[method,'Objective']
                expected=dp['scores'] if method=='EMDT-DP' else scores[method]
                assert np.array_equal(test_scores,expected)
                if method in ['EMDT','EMDT-DP']: verify_order(tree)
                if method=='EMDT-Cell':
                    splits=[]
                    for f in range(parts['Xfit'].shape[1]):
                        unique=np.unique(parts['Xfit'][:,f])
                        thresholds=np.percentile(parts['Xfit'][:,f],[20,40,60,80]) if len(unique)>5 else unique[:-1]
                        splits.append(sorted(set(thresholds.tolist())|{1.}))
                    cells=np.array(list(product(*splits))); shape=tuple(map(len,splits))
                    assert len(cells)==block.loc[method,'Cells']
                    cell_pred=predict(tree,cells).reshape(shape)
                    for f in range(len(shape)): assert np.all(np.diff(cell_pred,axis=f)>=0)
                check=audit(lambda X:predict(tree,X),parts,seed)
                for metric,value in check.items(): assert np.isclose(value,block.loc[method,metric],atol=1e-12)
                tree_count+=1
            assert block.loc['EMDT','Objective']==block.loc['EMDT-DP','Objective']
            calibration=json.loads((directory/'calibration.json').read_text())
            levels=np.array(calibration['output_scores']); assert levels[0]<=levels[1]
            assert np.allclose(levels[scores['EMDT'].astype(int)],scores['EMDT-Iso'])
            assert calibration['validation_rows']==len(parts['val'])
    for r in summary.itertuples():
        a=runs[(runs.Dataset==r.Dataset)&(runs.Method==r.Method)][r.Metric]
        lo,hi=ci(a)
        assert np.allclose([a.mean(),a.std(ddof=1),lo,hi],[r.Mean,r.SD,r.CI_L,r.CI_U],atol=1e-12)
    tex=(OUT.parent.parent/'main.tex').read_text(encoding='utf-8')
    assert '500-record' not in tex and 'archived' not in tex
    assert len(tex.split(r'\begin{abstract}')[1].split(r'\end{abstract}')[0].split())<200
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_audit.json']
    result=dict(test_metric_checks=metric_count,serialized_trees=tree_count,split_checks=40,
        summary_blocks=len(summary),source_hashes_verified=True,
        emdt_objective_equals_independent_dp_all_splits=True,
        hashes={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (OUT/'artifact_audit.json').write_text(json.dumps(result,indent=2))
    print('Verified',metric_count,'test metrics,',tree_count,'serialized trees, and',len(summary),'summary blocks.')

if __name__=='__main__': main()
