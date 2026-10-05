"""Exhaustive Boolean-domain expressivity analysis and CP-SAT cross-checks."""
from functools import lru_cache
from itertools import product
from pathlib import Path
import json
import platform
import sys

import numpy as np
import pandas as pd
import ortools
from emdt import ExactMonotonicTree

OUT = Path(__file__).resolve().parent
X = np.array(list(product([0., 1.], repeat=3)))
ALL = (1 << len(X)) - 1
MASKS = [sum(1 << i for i, x in enumerate(X) if x[j] == 1) for j in range(3)]


@lru_cache(None)
def subtree_functions(depth):
    """Track all structural leaf extrema, including unreachable leaves."""
    states = {(0, 0, 0), (ALL, 1, 1)}
    if depth:
        children = subtree_functions(depth - 1)
        for right_mask in MASKS:
            for low, low_min, low_max in children:
                for high, high_min, high_max in children:
                    if low_max <= high_min:
                        states.add(((low & (ALL ^ right_mask)) | (high & right_mask),
                                    min(low_min, high_min), max(low_max, high_max)))
    return frozenset(states)


def is_monotone(mask):
    return all(not (mask >> i & 1) or (mask >> k & 1)
               for i, a in enumerate(X) for k, b in enumerate(X) if np.all(a <= b))


def main():
    monotone = [m for m in range(ALL + 1) if is_monotone(m)]
    assert len(monotone) == 20
    counts = []
    for depth in [1, 2, 3]:
        represented = {s[0] for s in subtree_functions(depth)}
        assert all(is_monotone(m) for m in represented)
        counts.append(dict(Depth=depth, SubtreeFunctions=len(represented),
                           AllMonotoneFunctions=20, Excluded=20-len(represented)))
    records = []
    represented = {s[0] for s in subtree_functions(3)}
    for mask in monotone:
        y = np.array([mask >> i & 1 for i in range(len(X))])
        dp_error = min((mask ^ m).bit_count() for m in represented)
        for encoding in ['subtree', 'cells']:
            model = ExactMonotonicTree(max_depth=3, time_limit=15,
                                       monotonic_encoding=encoding)
            assert model.fit(X, y)
            assert model.fit_stats_['status'] == 'OPTIMAL'
            pred = model.predict(X)
            error = int(np.sum(pred != y))
            assert error == (dp_error if encoding == 'subtree' else 0)
            assert np.array_equal(pred, model.routing_predictions_[:len(X)])
            assert all(pred[i] <= pred[k] for i, a in enumerate(X)
                       for k, b in enumerate(X) if np.all(a <= b))
            records.append(dict(FunctionMask=mask, Encoding=encoding,
                                Errors=error, Status=model.fit_stats_['status']))

    # A partial monotone function with both signs of ungoverned dependence.
    y = (X[:, 0] * (1 - X[:, 1])).astype(int)
    partial = ExactMonotonicTree(max_depth=2, monotonic_features=[0],
                                monotonic_encoding='cells', time_limit=15)
    assert partial.fit(X, y) and np.array_equal(partial.predict(X), y)
    # Threshold endpoints and every interior candidate cell must route identically.
    train = np.linspace(0, 1, 11)[:, None]
    model = ExactMonotonicTree(max_depth=2, monotonic_encoding='cells', time_limit=15)
    assert model.fit(train, (train[:, 0] > 0.5).astype(int))
    assert np.array_equal(model.predict(model.cell_points_), model.routing_predictions_[len(train):])
    grid = np.linspace(0, 1, 1001)[:, None]
    assert np.all(np.diff(model.predict(grid)) >= 0)
    # UNKNOWN must not become an apparently certified constant tree.
    no_incumbent = ExactMonotonicTree(max_depth=3, time_limit=0)
    assert no_incumbent.fit(X, y) is False
    assert no_incumbent.fit_stats_['status'] == 'UNKNOWN'
    try:
        no_incumbent.predict(X)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Prediction without a feasible incumbent must fail')
    pd.DataFrame(counts).to_csv(OUT / 'model_class_counts.csv', index=False)
    pd.DataFrame(records).to_csv(OUT / 'model_class_fits.csv', index=False)
    majority = sum(1 << i for i, x in enumerate(X) if x.sum() >= 2)
    print(pd.DataFrame(counts).to_string(index=False))
    print(pd.DataFrame(records).query('FunctionMask == @majority').to_string(index=False))
    (OUT / 'model_class_environment.json').write_text(json.dumps(dict(
        python=sys.version, platform=platform.platform(), numpy=np.__version__,
        pandas=pd.__version__, ortools=ortools.__version__,
        domain='all eight binary profiles; all twenty monotone labelings',
        depth=3, workers=1, seed=42, solver_limit_seconds=15), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
