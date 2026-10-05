"""Figures and physical-unit case rules from saved artifacts only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report_full_cohort import OUT,ROOT,DATASETS,NAMES,ci

plt.rcParams.update({'font.size':16,'axes.labelsize':16,'xtick.labelsize':16,
                     'ytick.labelsize':16,'legend.fontsize':16,'pdf.fonttype':42})
FIG=ROOT/'figures'
def save(fig,name):
    fig.tight_layout(); fig.savefig(FIG/(name+'.png'),dpi=150,bbox_inches='tight')
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight'); plt.close(fig)

def main():
    b=pd.read_csv(ROOT/'experiment'/'boolean_expressivity.csv')
    fig,ax=plt.subplots(figsize=(6,4))
    ax.plot(b.Features,b.Fraction*100,'o-',color='#2166ac')
    ax.set(xlabel='Boolean variables',ylabel='Coverage (%)',ylim=(0,105),xticks=b.Features)
    ax.grid(alpha=.2); save(fig,'boolean_coverage')
    runs=pd.read_csv(OUT/'all_runs.csv')
    fig,ax=plt.subplots(figsize=(7,4.5)); x=np.arange(4)
    for j,(method,color) in enumerate([('EMDT','#2166ac'),('EMDT-DP','#d95f02'),('EMDT-Cell','#1b9e77')]):
        mean=[]; bounds=[]
        for ds in DATASETS:
            a=runs[(runs.Dataset==ds)&(runs.Method==method)].Time
            mean.append(a.mean()); bounds.append(ci(a))
        low,high=np.array(bounds).T; mean=np.array(mean)
        ax.errorbar(x+(j-1)*.08,mean,yerr=[mean-low,high-mean],fmt='o',label=method,color=color,capsize=3)
    ax.set(yscale='log',ylabel='Fit time (seconds)',xticks=x,xticklabels=[NAMES[s] for s in DATASETS])
    ax.legend(loc='upper right'); ax.grid(axis='y',alpha=.2); save(fig,'full_runtime')
    stress=pd.read_csv(OUT/'stress_runs.csv')
    for study,key,name,label in [('Size','N','dp_size','Total generated observations'),
                                 ('Depth','Depth','dp_depth','Maximum depth'),
                                 ('Features','Features','dp_features','Governed features')]:
        fig,ax=plt.subplots(figsize=(6,4))
        b=stress[(stress.Study==study)&(stress.Method=='EMDT-DP')]
        rows=[]
        for v,g in b.groupby(key):
            low,high=ci(g.Time); rows.append([v,g.Time.mean(),low,high])
        q=np.array(rows); ax.errorbar(q[:,0],q[:,1],yerr=[q[:,1]-q[:,2],q[:,3]-q[:,1]],fmt='o-',capsize=3)
        ax.set(xlabel=label,ylabel='Mean DP fit seconds',yscale='log')
        if study=='Size': ax.set_xscale('log')
        else: ax.set_xticks(q[:,0])
        ax.grid(alpha=.2); save(fig,name)
    fig,ax=plt.subplots(figsize=(7,4.5)); values=[]; labels=[]
    for ds in DATASETS:
        values.append(runs[(runs.Dataset==ds)&(runs.Method=='EMDT-DP')].Accuracy)
        labels.append(NAMES[ds])
    ax.boxplot(values,tick_labels=labels,showmeans=True); ax.set_ylabel('EMDT-DP test accuracy')
    ax.grid(axis='y',alpha=.2); save(fig,'full_stability')
    directory=OUT/'rules'; directory.mkdir(exist_ok=True); cases=[]
    for ds in DATASETS:
        job=OUT/'jobs'/f'{ds}_42'; tree=json.loads((job/'EMDT-DP_tree.json').read_text())
        scales=np.load(job/'split_and_scaling.npz'); rules=[]
        def labels(node):
            if str(node) in tree['leaves']: return {tree['leaves'][str(node)]}
            return labels(2*node)|labels(2*node+1)
        def walk(node,conditions):
            out=labels(node)
            if len(out)==1:
                rules.append(dict(conditions=conditions,label=next(iter(out)))); return
            f,t=tree['structure'][str(node)]; lo,hi=scales['lo'][f],scales['hi'][f]
            threshold=float(lo+t*(hi-lo)) if tree['directions'][f]>0 else float(hi-t*(hi-lo))
            low=' <= ' if tree['directions'][f]>0 else ' >= '
            high=' > ' if tree['directions'][f]>0 else ' < '
            walk(2*node,conditions+[f"{tree['features'][f]}{low}{threshold:.8g}"])
            walk(2*node+1,conditions+[f"{tree['features'][f]}{high}{threshold:.8g}"])
        walk(1,[])
        (directory/(ds+'.json')).write_text(json.dumps(dict(seed=42,rules=rules,
            note='Thresholds displayed to 8 significant digits; serialized normalized tree is authoritative.'),indent=2))
        text='; '.join((' AND '.join(r['conditions']) or 'all profiles')+f" -> {r['label']}" for r in rules)
        cases.append(NAMES[ds]+' & '+text.replace('_',r'\_').replace(' -> ',r' $\rightarrow$ ').replace(' >= ',r' $\ge$ ').replace(' <= ',r' $\le$ ')+r' \\')
    (ROOT/'case_rules.tex').write_text('\n'.join([
        r'\begin{table}[ht]\centering\small',
        r'\caption{Seed-42 DP case rules in raw units, with constant subtrees collapsed. Class 1 denotes the favorable target defined in the protocol. Thresholds are displayed to eight significant digits.}\label{tab:rules}',
        r'\begin{tabularx}{\linewidth}{@{}lZ@{}}\toprule Dataset & Rules \\\midrule',
        *cases,r'\bottomrule\end{tabularx}\end{table}']),encoding='utf-8')
    print((ROOT/'case_rules.tex').read_text())

if __name__=='__main__': main()
