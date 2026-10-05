"""Exact dynamic programming for fully governed binary-label subtree trees.

The subtree certificate implies one constant child at every internal node.
Depth is a maximum semantic depth; constant subtrees are padded to the requested
complete binary topology when exporting a tree. All candidate thresholds are
fixed from the original training data before exact row aggregation.
"""
from functools import lru_cache
import time
import numpy as np


class CascadeMonotonicTree:
    def __init__(self,max_depth=3):
        self.max_depth=max_depth
        self.solution_=None

    def fit(self,X,y):
        started=time.perf_counter()
        X=np.asarray(X,float); y=np.asarray(y,int)
        self.splits=[]
        for f in range(X.shape[1]):
            unique=np.unique(X[:,f])
            thresholds=np.percentile(X[:,f],np.linspace(20,80,4)) if len(unique)>5 else unique[:-1]
            self.splits.extend((f,float(t)) for t in thresholds)
        if not self.splits:
            raise ValueError('At least one candidate is required')
        B=np.column_stack([X[:,f]>t for f,t in self.splits])
        patterns,first,inverse=np.unique(B,axis=0,return_index=True,return_inverse=True)
        counts=np.zeros((len(first),2),int)
        np.add.at(counts,(inverse,y),1)
        zero=counts[:,0].tolist(); one=counts[:,1].tolist()
        full=(1<<len(first))-1
        right=[sum(1<<int(i) for i in np.flatnonzero(patterns[:,k])) for k in range(len(self.splits))]
        choices={}

        @lru_cache(None)
        def cost(mask):
            n0=n1=0
            while mask:
                bit=mask&-mask; i=bit.bit_length()-1
                n0+=zero[i]; n1+=one[i]; mask-=bit
            return n0,n1

        @lru_cache(None)
        def solve(mask,depth):
            n0,n1=cost(mask)
            best=min(n0,n1)
            chosen=('constant',int(n0<n1))
            if depth and best:
                for k,right_mask in enumerate(right):
                    high=mask&right_mask; low=mask^high
                    if not low or not high:
                        continue
                    # Left child constant 0; right child continues the cascade.
                    value=cost(low)[1]+solve(high,depth-1)
                    if value<best:
                        best=value; chosen=('split',k,'right',high)
                    # Right child constant 1; left child continues the cascade.
                    value=cost(high)[0]+solve(low,depth-1)
                    if value<best:
                        best=value; chosen=('split',k,'left',low)
            choices[mask,depth]=chosen
            return best

        objective=solve(full,self.max_depth)
        self.solution_=dict(structure={},leaves={})

        def constant(node,depth,label):
            if depth==0:
                self.solution_['leaves'][node]=label
            else:
                self.solution_['structure'][node]=self.splits[0]
                constant(2*node,depth-1,label); constant(2*node+1,depth-1,label)

        def export(mask,depth,node):
            choice=choices[mask,depth]
            if choice[0]=='constant':
                constant(node,depth,choice[1])
            else:
                _,k,side,child=choice
                self.solution_['structure'][node]=self.splits[k]
                if side=='right':
                    constant(2*node,depth-1,0); export(child,depth-1,2*node+1)
                else:
                    export(child,depth-1,2*node); constant(2*node+1,depth-1,1)

        export(full,self.max_depth,1)
        self.fit_stats_=dict(status='DP-OPTIMAL',objective=float(objective),best_bound=float(objective),
            relative_gap=0.,wall_time=time.perf_counter()-started,
            n_raw_training=len(X),n_training_patterns=len(first),states=solve.cache_info().currsize,
            cached_regions=cost.cache_info().currsize,fully_governed=True)
        self.n_splits=len(self.splits)
        return True

    def predict(self,X):
        if self.solution_ is None:
            raise RuntimeError('Fit the model first')
        X=np.asarray(X,float); predictions=[]
        last_internal=(1<<self.max_depth)-1
        for point in X:
            node=1
            while node<=last_internal:
                f,t=self.solution_['structure'][node]
                node=2*node+int(point[f]>t)
            predictions.append(self.solution_['leaves'][node])
        return np.array(predictions)
