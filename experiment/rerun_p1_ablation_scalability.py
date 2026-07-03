"""
P1-12 (fixed): Depth ablation with 10 seeds, mean±std.
Uses N=150, 3 features, 15s time limit to match real-experiment budget.
Depth-runtime trend is the key signal; N=150 is sufficient.

P1-13 (fixed): Scalability 5 repeats, 15s time limit each.

Run from learn13/experiment/ directory.
"""
import numpy as np
import pandas as pd
import time
from emdt import ExactMonotonicTree
from sklearn.model_selection import train_test_split

# ─── Data generators ─────────────────────────────────────────────────────────
def make_credit_like(n=150, noise=0.15, seed=42):
    """Small credit-like 3-feature monotone dataset."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, 3))           # 3 features, smaller N
    score = 0.5*X[:,0] + 0.3*X[:,1] + 0.2*X[:,2]
    y = (score > 0.5).astype(int)
    y[rng.uniform(size=n) < noise] ^= 1
    return X, y

def make_linear(n, d=3, noise=0.1, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, d))
    p = 1 / (1 + np.exp(-3*(X.sum(1) - d/2)))
    y = (rng.uniform(size=n) < p).astype(int)
    y[rng.uniform(size=n) < noise] ^= 1
    return X, y

TIME_LIMIT = 15          # seconds — consistent with real-data experiments

# ── P1-12: Depth ablation, 10 seeds ──────────────────────────────────────────
print("=" * 60)
print("P1-12: Depth ablation, 10 seeds (N=150, 3 features, 15s limit)")
print("=" * 60)

SEEDS_10 = list(range(42, 52))
rows_abl = []

for depth in [2, 3, 4]:
    for seed in SEEDS_10:
        X, y = make_credit_like(n=150, noise=0.15, seed=seed)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y)
        t0 = time.time()
        clf = ExactMonotonicTree(max_depth=depth, time_limit=TIME_LIMIT,
                                 monotonic=True, random_seed=seed)
        ok = clf.fit(Xtr, ytr)
        elapsed = time.time() - t0
        tr_acc = float((clf.predict(Xtr) == ytr).mean()) if ok else float('nan')
        te_acc = float((clf.predict(Xte) == yte).mean()) if ok else float('nan')
        status = clf.fit_stats_.get('status', 'N/A')
        gap    = clf.fit_stats_.get('relative_gap', float('nan'))
        rows_abl.append(dict(depth=depth, seed=seed, time=elapsed,
                              train_acc=tr_acc, test_acc=te_acc,
                              status=status, gap=gap))
        print(f"  depth={depth} seed={seed} t={elapsed:.2f}s "
              f"tr={tr_acc:.3f} te={te_acc:.3f} [{status}]", flush=True)

df_abl = pd.DataFrame(rows_abl)
df_abl.to_csv('depth_ablation_10seeds.csv', index=False)

abl_sum = (df_abl.groupby('depth')
           .agg(time_mean=('time','mean'), time_std=('time','std'),
                train_acc_mean=('train_acc','mean'), train_acc_std=('train_acc','std'),
                test_acc_mean=('test_acc','mean'),  test_acc_std=('test_acc','std'))
           .reset_index())
abl_sum.to_csv('depth_ablation_10seeds_summary.csv', index=False)

print("\n=== DEPTH ABLATION SUMMARY (10 seeds, N=150, 15s) ===")
print(abl_sum.to_string(index=False))

# ── P1-13: Scalability, 5 repeats per N ──────────────────────────────────────
print("\n" + "=" * 60)
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
print("\nDone: depth_ablation_10seeds.csv, scalability_5reps.csv")
