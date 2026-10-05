from itertools import product
import numpy as np
from cascade_dp import CascadeMonotonicTree
from emdt import ExactMonotonicTree
from analyze_model_class import subtree_functions


def main():
    X=np.array(list(product([0.,1.],repeat=3)))
    for mask in range(256):
        y=np.array([mask>>i&1 for i in range(8)])
        model=CascadeMonotonicTree(max_depth=3)
        model.fit(X,y)
        minimum=min((mask^f).bit_count() for f,_,_ in subtree_functions(3))
        assert model.fit_stats_['objective']==minimum
        assert np.sum(model.predict(X)!=y)==minimum
    rng=np.random.default_rng(818)
    for depth in [1,2,3]:
        for trial in range(4):
            continuous=rng.random((40,3)); y=rng.integers(0,2,40)
            dp=CascadeMonotonicTree(depth); dp.fit(continuous,y)
            cp=ExactMonotonicTree(max_depth=depth,time_limit=30,compress_rows=True,constant_hint=True)
            assert cp.fit(continuous,y) and cp.fit_stats_['status']=='OPTIMAL'
            assert cp.fit_stats_['objective']==dp.fit_stats_['objective']
            assert np.sum(dp.predict(continuous)!=y)==dp.fit_stats_['objective']
    print('All 256 Boolean labelings match exhaustive optima; 12 continuous-data objectives match certified CP-SAT optima.')


if __name__=='__main__':
    main()
