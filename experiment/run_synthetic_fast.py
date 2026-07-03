"""
P1-11 (fast version): Synthetic benchmark re-run with 30 seeds.
Uses joblib to run in parallel across CPU cores.
"""
import numpy as np
import pandas as pd
import time
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from emdt import ExactMonotonicTree

def make_linear(n, d, noise, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, d))
    p = 1 / (1 + np.exp(-3*(X.sum(1) - d/2)))
    y = (rng.uniform(size=n) < p).astype(int)
    y[rng.uniform(size=n) < noise] ^= 1
    return X, y

def make_xor(n, d, noise, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, d))
    p = np.logical_xor(X[:,0] > 0.5, X[:,1] > 0.5).astype(float)
    y = (rng.uniform(size=n) < p).astype(int)
    y[rng.uniform(size=n) < noise] ^= 1
    return X, y

def make_poly(n, d, noise, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, d))
    p = 1 / (1 + np.exp(-5*(X[:,0]*X[:,1] - 0.25)))
    y = (rng.uniform(size=n) < p).astype(int)
    y[rng.uniform(size=n) < noise] ^= 1
    return X, y

def run_single_seed(dataset_name, noise, seed):
    N, D = 200, 3
    if dataset_name == 'Linear':
        X, y = make_linear(N, D, noise, seed)
    elif dataset_name == 'XOR':
        X, y = make_xor(N, D, noise, seed)
    else:
        X, y = make_poly(N, D, noise, seed)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    
    # CART
    t0 = time.time()
    cart = DecisionTreeClassifier(max_depth=3, random_state=seed)
    cart.fit(Xtr, ytr)
    t_cart = time.time() - t0
    acc_cart = cart.score(Xte, yte)
    
    # EMDT
    t0 = time.time()
    emdt = ExactMonotonicTree(max_depth=3, time_limit=15, monotonic=True, random_seed=seed, num_workers=1)
    emdt.fit(Xtr, ytr)
    t_emdt = time.time() - t0
    acc_emdt = np.mean(emdt.predict(Xte) == yte)
    
    return {
        'Dataset': dataset_name, 'Noise': noise, 'Seed': seed,
        'CART_acc': acc_cart, 'CART_time': t_cart,
        'EMDT_acc': acc_emdt, 'EMDT_time': t_emdt,
        'EMDT_status': emdt.fit_stats_.get('status', 'N/A')
    }

seeds = list(range(42, 72))
datasets = ['Linear', 'XOR', 'Polynomial']
noises = [0.1, 0.25]

print(f"Starting 30-seed synthetic run in parallel...")
tasks = []
for d in datasets:
    for n in noises:
        for s in seeds:
            tasks.append((d, n, s))

results = Parallel(n_jobs=-1, verbose=10)(delayed(run_single_seed)(*args) for args in tasks)

df = pd.DataFrame(results)
df.to_csv('synthetic_30seeds_raw.csv', index=False)

summary = df.groupby(['Dataset', 'Noise']).agg({
    'CART_acc': ['mean', 'std'],
    'EMDT_acc': ['mean', 'std'],
    'CART_time': 'mean',
    'EMDT_time': 'mean'
}).reset_index()

summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
summary.to_csv('synthetic_30seeds_summary.csv', index=False)
print(summary)
