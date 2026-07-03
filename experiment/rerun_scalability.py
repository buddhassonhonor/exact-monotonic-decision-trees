"""
P1-13 (fixed): Scalability 5 repeats, 15s time limit each.
"""
import numpy as np
import pandas as pd
import time
from emdt import ExactMonotonicTree
from sklearn.model_selection import train_test_split

def make_linear(n, d=3, noise=0.1, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, d))
    p = 1 / (1 + np.exp(-3*(X.sum(1) - d/2)))
    y = (rng.uniform(size=n) < p).astype(int)
    y[rng.uniform(size=n) < noise] ^= 1
    return X, y

TIME_LIMIT = 15

print("=" * 60)
print("P1-13: Scalability 5 repeats per N (linear data, depth=3, 15s)")
print("=" * 60)

N_VALUES = [100, 200, 300, 400]
N_REPS   = 5
rows_scal = []

for N in N_VALUES:
    for rep in range(N_REPS):
        seed = 42 + rep
        X, y = make_linear(n=N, noise=0.1, seed=seed)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y)
        t0 = time.time()
        clf = ExactMonotonicTree(max_depth=3, time_limit=TIME_LIMIT,
                                 monotonic=True, random_seed=seed)
        clf.fit(Xtr, ytr)
        elapsed = time.time() - t0
        status = clf.fit_stats_.get('status', 'N/A')
        rows_scal.append(dict(N=N, rep=rep, seed=seed, time=elapsed, status=status))
        print(f"  N={N} rep={rep} t={elapsed:.3f}s [{status}]", flush=True)

df_scal = pd.DataFrame(rows_scal)
df_scal.to_csv('scalability_5reps.csv', index=False)

scal_sum = (df_scal.groupby('N')
            .agg(time_mean=('time','mean'), time_std=('time','std'))
            .reset_index())
scal_sum.to_csv('scalability_5reps_summary.csv', index=False)

print("\n=== SCALABILITY SUMMARY (5 reps, 15s limit) ===")
print(scal_sum.to_string(index=False))
