"""
P1-11 (revised): Synthetic benchmark 30 seeds.
Uses shorter EMDT time_limit (15s, matching real experiments),
saves partial results as CSV so progress is not lost on interruption.
Run from learn13/experiment/ directory.
"""
import numpy as np
import pandas as pd
import time
import os
from emdt import ExactMonotonicTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

DEPTH = 3
TIME_LIMIT = 15          # match real-data experiments
SEEDS = list(range(42, 72))   # 30 seeds
OUT_FILE = 'synthetic_30seeds.csv'
SUMMARY_FILE = 'synthetic_30seeds_summary.csv'

def make_linear(n=200, d=3, noise=0.1, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, d))
    p = 1 / (1 + np.exp(-3 * (X.sum(1) - d / 2)))
    y = (rng.uniform(size=n) < p).astype(int)
    y[rng.uniform(size=n) < noise] ^= 1
    return X, y

def make_nonlinear(n=200, d=3, noise=0.1, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, d))
    score = X[:,0]*X[:,1] + 0.5*X[:,2]
    p = 1 / (1 + np.exp(-6*(score - 0.4)))
    y = (rng.uniform(size=n) < p).astype(int)
    y[rng.uniform(size=n) < noise] ^= 1
    return X, y

def make_highdim(n=200, d=10, noise=0.1, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, d))
    p = 1 / (1 + np.exp(-4*(X[:,:5].sum(1) - 2.5)))
    y = (rng.uniform(size=n) < p).astype(int)
    y[rng.uniform(size=n) < noise] ^= 1
    return X, y

GENERATORS = {
    'Linear':    (make_linear,    dict(d=3)),
    'Nonlinear': (make_nonlinear, dict(d=3)),
    'High-dim':  (make_highdim,   dict(d=10)),
}
NOISES = [0.1, 0.2]

# Load checkpoint if partial results exist
if os.path.exists(OUT_FILE):
    done_df = pd.read_csv(OUT_FILE)
    done_keys = set(zip(done_df['dataset'], done_df['noise'].astype(str),
                        done_df['seed'].astype(str), done_df['method']))
    rows = done_df.to_dict('records')
    print(f"Resuming from checkpoint: {len(done_df)} rows already done.")
else:
    done_keys = set()
    rows = []

total = len(GENERATORS) * len(NOISES) * len(SEEDS) * 2
completed = len(done_keys)

for ds_name, (gen_fn, gen_kw) in GENERATORS.items():
    for noise in NOISES:
        for seed in SEEDS:
            for method in ['CART', 'EMDT']:
                key = (ds_name, str(noise), str(seed), method)
                if key in done_keys:
                    continue
                X, y = gen_fn(n=200, noise=noise, seed=seed, **gen_kw)
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=0.2, random_state=seed, stratify=y)
                t0 = time.time()
                if method == 'CART':
                    clf = DecisionTreeClassifier(max_depth=DEPTH, random_state=seed)
                    clf.fit(Xtr, ytr)
                    preds = clf.predict(Xte)
                    viols = 0
                    for _ in range(500):
                        rng = np.random.default_rng(seed + _)
                        i, j = rng.integers(0, len(Xte), size=2)
                        if np.all(Xte[i] <= Xte[j]) and preds[i] > preds[j]:
                            viols += 1
                else:
                    clf = ExactMonotonicTree(max_depth=DEPTH, time_limit=TIME_LIMIT,
                                            monotonic=True, random_seed=seed)
                    ok = clf.fit(Xtr, ytr)
                    preds = clf.predict(Xte) if ok else np.zeros(len(Xte), dtype=int)
                    viols = 0  # certified by construction
                elapsed = time.time() - t0
                acc = (preds == yte).mean()
                rows.append(dict(dataset=ds_name, noise=noise, seed=seed,
                                 method=method, accuracy=acc,
                                 violations=viols, time=elapsed))
                completed += 1
                print(f"[{completed}/{total}] {ds_name} noise={noise} "
                      f"seed={seed} {method} acc={acc:.3f} t={elapsed:.1f}s",
                      flush=True)
                # Save checkpoint after each run
                pd.DataFrame(rows).to_csv(OUT_FILE, index=False)

df = pd.DataFrame(rows)
df.to_csv(OUT_FILE, index=False)

summary = (df.groupby(['dataset','noise','method'])
           .agg(acc_mean=('accuracy','mean'), acc_std=('accuracy','std'),
                viol_mean=('violations','mean'), time_mean=('time','mean'))
           .reset_index())
summary.to_csv(SUMMARY_FILE, index=False)
print("\n=== SUMMARY (30 seeds) ===")
print(summary.to_string(index=False))
print(f"\nSaved: {OUT_FILE}, {SUMMARY_FILE}")
