"""Verify exact aggregation under conflicting labels and identical split behavior."""
from itertools import product
import numpy as np
from emdt import ExactMonotonicTree


def main():
    X = np.repeat(np.array(list(product([0., 1.], repeat=3))), 3, axis=0)
    rng = np.random.default_rng(987)
    for encoding, governed in [('subtree',None),('cells',None),('cells',[0])]:
        for repeat in range(4):
            y = rng.integers(0,2,len(X))
            objectives=[]
            for compress in [False, True]:
                model=ExactMonotonicTree(max_depth=3, time_limit=15,
                    monotonic_encoding=encoding, monotonic_features=governed,
                    compress_rows=compress, constant_hint=True)
                assert model.fit(X,y)
                assert model.fit_stats_['status']=='OPTIMAL'
                obj=model.fit_stats_['objective']
                assert obj == np.sum(model.predict(X)!=y)
                assert model.class_counts_.sum()==len(X)
                objectives.append(obj)
            assert objectives[0]==objectives[1]
    print('24 optimal solves: compressed and uncompressed objectives agree; weighted errors match raw predictions.')


if __name__=='__main__':
    main()
