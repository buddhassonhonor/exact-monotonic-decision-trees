"""Exhaustive monotone Boolean functions and structural-class counts up to five bits."""
from functools import lru_cache
from pathlib import Path
import json
import time
import pandas as pd


@lru_cache(None)
def monotone_functions(n):
    if n==0:
        return frozenset([0,1])
    previous=monotone_functions(n-1)
    half=1<<(n-1)
    return frozenset(low | (high<<half) for low in previous for high in previous
                     if low & ~high == 0)


def branch_mask(subfunction,n,j,branch):
    shift=n-1-j; low_mask=(1<<shift)-1
    result=0
    for old_index in range(1<<(n-1)):
        if subfunction>>old_index & 1:
            new_index=((old_index>>shift)<<(shift+1)) | (branch<<shift) | (old_index&low_mask)
            result|=1<<new_index
    return result


@lru_cache(None)
def structural_functions(n,depth):
    all_mask=(1<<(1<<n))-1
    results={0,all_mask}
    if n==0 or depth==0:
        return frozenset(results)
    for j in range(n):
        feature_mask=branch_mask((1<<(1<<(n-1)))-1,n,j,1)
        for child in structural_functions(n-1,depth-1):
            results.add(branch_mask(child,n,j,1))       # x_j AND child
            results.add(branch_mask(child,n,j,0)|feature_mask) # x_j OR child
    return frozenset(results)


def main():
    started=time.perf_counter(); counts=[]
    for n in range(1,6):
        mono=monotone_functions(n); structural=structural_functions(n,n)
        assert structural<=mono
        assert structural_functions(n,n+1)==structural
        majority=sum(1<<i for i in range(1<<n) if i.bit_count() >= n//2+1)
        if n>=3:
            assert majority in mono and majority not in structural
        counts.append(dict(Features=n,Depth=n,MonotoneFunctions=len(mono),
                           StructuralFunctions=len(structural),Excluded=len(mono)-len(structural),
                           Fraction=len(structural)/len(mono)))
    assert len(monotone_functions(3))==20 and len(structural_functions(3,3))==19
    assert [len(structural_functions(3,d)) for d in [1,2,3]]==[5,11,19]
    out=Path(__file__).resolve().parent
    df=pd.DataFrame(counts)
    df.to_csv(out/'boolean_expressivity.csv',index=False)
    (out/'boolean_expressivity_protocol.json').write_text(json.dumps(dict(
        domain='All Boolean profiles',features=list(range(1,6)),depth='equal to feature count',
        monotone_generation='Recursive ordered cofactors, exhaustive',
        structural_generation='Recursive AND/OR chain with one constant root branch',
        solver_independent=True,elapsed_seconds=time.perf_counter()-started),indent=2))
    print(df.to_string(index=False))


if __name__=='__main__':
    main()
