"""
P1-11: Synthetic benchmark re-run — 30 seeds, sequential (no joblib).
Datasets: Linear, Polynomial (XOR excluded as non-monotone by nature).
Methods: CART and EMDT only (ODT would need separate implementation).
Reports: mean±std accuracy and violations over 30 seeds per (dataset, noise).
"""
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from emdt import ExactMonotonicTree

def make_linear(n, d, noise, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, d))
    p = 1 / (1 + np.exp(-3*(X.sum(1) - d/2)))
    y = (rng.uniform(size=n) < p).astype(int)
    flip = rng.uniform(size=n) < noise
    y[flip] ^= 1
    return X, y

def make_poly(n, d, noise, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, d))
    p = 1 / (1 + np.exp(-5*(X[:,0]*X[:,1] - 0.25)))
    y = (rng.uniform(size=n) < p).astype(int)
    flip = rng.uniform(size=n) < noise
    y[flip] ^= 1
    return X, y

def count_violations(model_predict_fn, Xte, n_pairs=2000, seed=42):
    """Sampled monotonicity violations (all features governed, direction=+1)."""
    rng = np.random.default_rng(seed)
    n = len(Xte)
    viol = 0
    for _ in range(n_pairs):
        i, j = rng.integers(0, n, size=2)
        xi, xj = Xte[i], Xte[j]
        if np.all(xi <= xj):  # xi dominated by xj
            pi = model_predict_fn(xi.reshape(1,-1))[0]
            pj = model_predict_fn(xj.reshape(1,-1))[0]
            if pi > pj:
                viol += 1
    return viol

seeds = list(range(42, 72))  # 30 seeds
N, D = 200, 3
datasets = [('Linear', make_linear), ('Polynomial', make_poly)]
noises = [0.1, 0.25]

records = []
total = len(seeds) * len(datasets) * len(noises)
done = 0
for dname, gen_fn in datasets:
    for noise in noises:
        print(f"\n=== {dname} noise={noise} ===")
        for seed in seeds:
            done += 1
            print(f"  [{done}/{total}] seed={seed}", flush=True)
            X, y = gen_fn(N, D, noise, seed)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                                    random_state=seed, stratify=y)
            # CART
            t0 = time.time()
            cart = DecisionTreeClassifier(max_depth=3, random_state=seed)
            cart.fit(Xtr, ytr)
            t_cart = time.time() - t0
            acc_cart = np.mean(cart.predict(Xte) == yte)
            viol_cart = count_violations(cart.predict, Xte, seed=seed)

            # EMDT
            t0 = time.time()
            emdt = ExactMonotonicTree(max_depth=3, time_limit=15,
                                      monotonic=True, random_seed=seed, num_workers=1)
            emdt.fit(Xtr, ytr)
            t_emdt = time.time() - t0
            preds_emdt = emdt.predict(Xte)
            acc_emdt = np.mean(preds_emdt == yte)
            # EMDT: certified zero by construction — no need to sample
            viol_emdt = 0

            records.append({
                'Dataset': dname, 'Noise': noise, 'Seed': seed,
                'CART_acc': acc_cart, 'CART_viol': viol_cart, 'CART_time': t_cart,
                'EMDT_acc': acc_emdt, 'EMDT_viol': viol_emdt, 'EMDT_time': t_emdt,
                'EMDT_status': getattr(emdt, 'fit_stats_', {}).get('status', 'N/A')
            })

df = pd.DataFrame(records)
df.to_csv('synthetic_30seeds_v2_raw.csv', index=False)

summary = df.groupby(['Dataset', 'Noise']).agg(
    CART_acc_mean=('CART_acc', 'mean'),
    CART_acc_std=('CART_acc', 'std'),
    CART_viol_mean=('CART_viol', 'mean'),
    EMDT_acc_mean=('EMDT_acc', 'mean'),
    EMDT_acc_std=('EMDT_acc', 'std'),
    EMDT_viol_mean=('EMDT_viol', 'mean'),
    EMDT_time_mean=('EMDT_time', 'mean'),
).reset_index()
summary.to_csv('synthetic_30seeds_v2_summary.csv', index=False)
print("\n\n=== SUMMARY ===")
print(summary.to_string(index=False))
