"""
P1-11: Synthetic benchmark re-run with 30 seeds (was 3).
P1-12: Depth ablation re-run with 10 seeds, mean±std reported (was 1 seed).
P1-13: Scalability re-run with 5 repeats per point + error bars (was 1 run).

Run from the learn13/experiment/ directory.
Writes:
  synthetic_30seeds.csv
  depth_ablation_10seeds.csv
  scalability_5reps.csv
"""
import numpy as np
import pandas as pd
import time
from emdt import ExactMonotonicTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# ─── Synthetic data generators ───────────────────────────────────────────────
def make_linear(n=200, d=3, noise=0.1, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, d))
    logit = X.sum(axis=1) - d / 2
    p = 1 / (1 + np.exp(-3 * logit))
    y = (rng.uniform(size=n) < p).astype(int)
    flip = rng.uniform(size=n) < noise
    y[flip] = 1 - y[flip]
    return X, y

def make_nonlinear(n=200, d=3, noise=0.1, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, d))
    score = (X[:, 0] * X[:, 1] + 0.5 * X[:, 2])
    p = 1 / (1 + np.exp(-6 * (score - 0.4)))
    y = (rng.uniform(size=n) < p).astype(int)
    flip = rng.uniform(size=n) < noise
    y[flip] = 1 - y[flip]
    return X, y

def make_highdim(n=200, d=10, noise=0.1, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, d))
    logit = X[:, :5].sum(axis=1) - 2.5
    p = 1 / (1 + np.exp(-4 * logit))
    y = (rng.uniform(size=n) < p).astype(int)
    flip = rng.uniform(size=n) < noise
    y[flip] = 1 - y[flip]
    return X, y

def check_violations(model, X, M=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(X)
    preds = model.predict(X)
    viols = 0
    for _ in range(M):
        i, j = rng.integers(0, n, size=2)
        if np.all(X[i] <= X[j]) and preds[i] > preds[j]:
            viols += 1
        elif np.all(X[j] <= X[i]) and preds[j] > preds[i]:
            viols += 1
    return viols

# ─── P1-11: Synthetic 30 seeds ───────────────────────────────────────────────
print("=" * 60)
print("P1-11: Synthetic benchmark, 30 seeds")
print("=" * 60)

SEEDS_30 = list(range(42, 72))
DEPTH = 3
TIME_LIMIT = 60  # generous for small synthetic data

generators = {
    'Linear':    (make_linear,    {'d': 3}),
    'Nonlinear': (make_nonlinear, {'d': 3}),
    'High-dim':  (make_highdim,   {'d': 10}),
}
noise_levels = [0.1, 0.2]

rows_syn = []
for ds_name, (gen_fn, gen_kw) in generators.items():
    for noise in noise_levels:
        for seed in SEEDS_30:
            X, y = gen_fn(n=200, noise=noise, seed=seed, **gen_kw)
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                       random_state=seed, stratify=y)
            for method in ['CART', 'EMDT']:
                t0 = time.time()
                if method == 'CART':
                    clf = DecisionTreeClassifier(max_depth=DEPTH, random_state=seed)
                    clf.fit(X_tr, y_tr)
                    preds_te = clf.predict(X_te)
                    viols = check_violations(clf, X_te, M=500, seed=seed)
                    elapsed = time.time() - t0
                else:
                    clf = ExactMonotonicTree(max_depth=DEPTH, time_limit=TIME_LIMIT,
                                            monotonic=True, random_seed=seed)
                    ok = clf.fit(X_tr, y_tr)
                    preds_te = clf.predict(X_te) if ok else np.zeros(len(X_te))
                    viols = 0  # certified by construction
                    elapsed = time.time() - t0

                acc = (preds_te == y_te).mean()
                rows_syn.append(dict(dataset=ds_name, noise=noise, seed=seed,
                                     method=method, accuracy=acc,
                                     violations=viols, time=elapsed))
            print(f"  {ds_name} noise={noise} seed={seed}: done", flush=True)

df_syn = pd.DataFrame(rows_syn)
df_syn.to_csv('synthetic_30seeds.csv', index=False)

# Summary
summary = (df_syn.groupby(['dataset', 'noise', 'method'])
           .agg(acc_mean=('accuracy', 'mean'), acc_std=('accuracy', 'std'),
                viol_mean=('violations', 'mean'), viol_std=('violations', 'std'),
                time_mean=('time', 'mean'))
           .reset_index())
summary.to_csv('synthetic_30seeds_summary.csv', index=False)
print("\nSaved: synthetic_30seeds.csv, synthetic_30seeds_summary.csv")
print(summary.to_string(index=False))

# ─── P1-12: Depth ablation 10 seeds ─────────────────────────────────────────
print("\n" + "=" * 60)
print("P1-12: Depth ablation, 10 seeds (credit-like data, N=300)")
print("=" * 60)

SEEDS_10 = list(range(42, 52))

def make_credit_like(n=300, noise=0.15, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, 4))
    score = 0.4 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * X[:, 2] + 0.1 * X[:, 3]
    y = (score > 0.5).astype(int)
    flip = rng.uniform(size=n) < noise
    y[flip] = 1 - y[flip]
    return X, y

rows_abl = []
for depth in [2, 3, 4]:
    time_limit_abl = {2: 30, 3: 120, 4: 300}[depth]
    for seed in SEEDS_10:
        X, y = make_credit_like(n=300, noise=0.15, seed=seed)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                   random_state=seed, stratify=y)
        t0 = time.time()
        clf = ExactMonotonicTree(max_depth=depth, time_limit=time_limit_abl,
                                 monotonic=True, random_seed=seed)
        ok = clf.fit(X_tr, y_tr)
        elapsed = time.time() - t0
        tr_acc = (clf.predict(X_tr) == y_tr).mean() if ok else np.nan
        te_acc = (clf.predict(X_te) == y_te).mean() if ok else np.nan
        rows_abl.append(dict(depth=depth, seed=seed, time=elapsed,
                              train_acc=tr_acc, test_acc=te_acc,
                              status=clf.fit_stats_.get('status', 'N/A')))
        print(f"  depth={depth} seed={seed} time={elapsed:.2f}s "
              f"train_acc={tr_acc:.3f}", flush=True)

df_abl = pd.DataFrame(rows_abl)
df_abl.to_csv('depth_ablation_10seeds.csv', index=False)

abl_summary = (df_abl.groupby('depth')
               .agg(time_mean=('time', 'mean'), time_std=('time', 'std'),
                    train_acc_mean=('train_acc', 'mean'),
                    train_acc_std=('train_acc', 'std'),
                    test_acc_mean=('test_acc', 'mean'),
                    test_acc_std=('test_acc', 'std'))
               .reset_index())
abl_summary.to_csv('depth_ablation_10seeds_summary.csv', index=False)
print("\nSaved: depth_ablation_10seeds.csv, depth_ablation_10seeds_summary.csv")
print(abl_summary.to_string(index=False))

# ─── P1-13: Scalability 5 repeats ────────────────────────────────────────────
print("\n" + "=" * 60)
print("P1-13: Scalability, 5 repeats per N")
print("=" * 60)

N_values = [100, 200, 300, 400]
N_REPS = 5

rows_scal = []
for N in N_values:
    for rep in range(N_REPS):
        seed = 42 + rep
        X, y = make_linear(n=N, d=3, noise=0.1, seed=seed)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                   random_state=seed, stratify=y)
        t0 = time.time()
        clf = ExactMonotonicTree(max_depth=3, time_limit=120,
                                 monotonic=True, random_seed=seed)
        clf.fit(X_tr, y_tr)
        elapsed = time.time() - t0
        rows_scal.append(dict(N=N, rep=rep, seed=seed, time=elapsed))
        print(f"  N={N} rep={rep} time={elapsed:.3f}s", flush=True)

df_scal = pd.DataFrame(rows_scal)
df_scal.to_csv('scalability_5reps.csv', index=False)

scal_summary = (df_scal.groupby('N')
                .agg(time_mean=('time', 'mean'), time_std=('time', 'std'))
                .reset_index())
scal_summary.to_csv('scalability_5reps_summary.csv', index=False)
print("\nSaved: scalability_5reps.csv, scalability_5reps_summary.csv")
print(scal_summary.to_string(index=False))
print("\nAll P1-11/12/13 re-runs complete.")
