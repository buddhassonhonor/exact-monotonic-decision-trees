"""Paired solver-only timings with/without memoization, same recurrence."""
from functools import lru_cache
import time
import pandas as pd
import numpy as np
from run_exact_stress import data,OUT
from cascade_dp import CascadeMonotonicTree

def main():
    records=[]
    for seed in range(42,47):
        parts=data('Linear',1000,3,.1,seed); X,y=parts['Xfit'],parts['yfit']
        base=CascadeMonotonicTree(3); base.fit(X,y)
        B=np.column_stack([X[:,f]>t for f,t in base.splits])
        patterns,inv=np.unique(B,axis=0,return_inverse=True)
        counts=np.zeros((len(patterns),2),int); np.add.at(counts,(inv,y),1)
        rights=[sum(1<<int(j) for j in np.flatnonzero(patterns[:,k])) for k in range(len(base.splits))]
        @lru_cache(None)
        def cost(mask):
            n0=n1=0
            while mask:
                bit=mask&-mask; j=bit.bit_length()-1
                n0+=int(counts[j,0]); n1+=int(counts[j,1]); mask-=bit
            return n0,n1
        for cached in [False,True]:
            cost.cache_clear(); calls=[0]
            def recurrence(mask,h):
                calls[0]+=1; n0,n1=cost(mask); best=min(n0,n1)
                if h and best:
                    for right in rights:
                        high=mask&right; low=mask^high
                        if low and high:
                            best=min(best,cost(low)[1]+solve(high,h-1),cost(high)[0]+solve(low,h-1))
                return best
            solve=lru_cache(None)(recurrence) if cached else recurrence
            start=time.perf_counter(); optimum=solve((1<<len(patterns))-1,3); elapsed=time.perf_counter()-start
            assert optimum==base.fit_stats_['objective']
            records.append(dict(Seed=seed,Cached=cached,Seconds=elapsed,Calls=calls[0],Objective=optimum))
    frame=pd.DataFrame(records); frame.to_csv(OUT/'cache_ablation.csv',index=False)
    print(frame.groupby('Cached')[['Seconds','Calls']].mean().to_string())

if __name__=='__main__': main()
