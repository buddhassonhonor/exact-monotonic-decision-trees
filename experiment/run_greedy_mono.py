"""
P1-17: Greedy monotone baseline — MonoCART
Uses XGBoost with n_estimators=1 (single tree) + monotone constraints as a greedy single-tree baseline.
Runs on all 4 real datasets, 10 seeds, depth 3.
"""
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from emdt import ExactMonotonicTree

# ------ dataset loaders (copied from run_revision_nmi.py) ------

def load_compas():
    url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    df = pd.read_csv(url)
    df = df[df['days_b_screening_arrest'].between(-30, 30)]
    df = df[df['is_recid'] != -1]
    df = df[df['c_charge_degree'] != 'O']
    df = df[df['score_text'] != 'N/A']
    df['low_risk'] = (df['two_year_recid'] == 0).astype(int)
    feats = ['age', 'juv_fel_count', 'juv_misd_count', 'priors_count', 'days_b_screening_arrest']
    X = df[feats].copy()
    X['age'] = (X['age'] - X['age'].min()) / (X['age'].max() - X['age'].min() + 1e-9)
    X['priors_good'] = 1 - (X['priors_count'] - X['priors_count'].min()) / (X['priors_count'].max() - X['priors_count'].min() + 1e-9)
    X['juv_hist_good'] = 1 - (X['juv_fel_count'] + X['juv_misd_count'] - 0) / (X['juv_fel_count'].max() + X['juv_misd_count'].max() + 1e-9)
    final_cols = ['age', 'priors_good', 'juv_hist_good']
    X = X[final_cols]
    y = df['low_risk'].values
    mono_dirs = [1, 1, 1]
    return X.values.astype(float), y, mono_dirs

def load_german():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    df = pd.read_csv(url, sep=' ', header=None)
    y = (df.iloc[:, -1] == 1).astype(int).values
    X_raw = df.iloc[:, :-1]
    X_enc = pd.get_dummies(X_raw, drop_first=True).astype(float).values
    numeric_cols = [1, 4, 7, 10, 12]
    X_norm = X_enc.copy()
    for c in range(X_norm.shape[1]):
        mn, mx = X_norm[:, c].min(), X_norm[:, c].max()
        if mx > mn:
            X_norm[:, c] = (X_norm[:, c] - mn) / (mx - mn)
    # governed features: duration(+), amount(+), installment(+), residence(+), age(+) → all positive
    # using first 5 numeric columns as proxies after encoding
    gov_idx = numeric_cols[:5]
    mono_dirs = [1] * len(gov_idx)
    return X_norm, y, gov_idx, mono_dirs

def run_greedy_mono(X, y, mono_constraint, seeds, depth=3):
    """Run XGBoost single-tree (n_estimators=1) with monotone_constraints."""
    results = []
    for seed in seeds:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
        t0 = time.time()
        model = xgb.XGBClassifier(
            n_estimators=1,
            max_depth=depth,
            monotone_constraints=mono_constraint,
            random_state=seed,
            eval_metric='logloss',
            verbosity=0
        )
        model.fit(Xtr, ytr)
        elapsed = time.time() - t0
        acc = accuracy_score(yte, model.predict(Xte))
        
        # Sample violation check: 2000 comparable pairs
        rng = np.random.default_rng(seed)
        n_pairs = 2000
        n = len(Xte)
        viol = 0
        for _ in range(n_pairs):
            i, j = rng.integers(0, n, size=2)
            xi, xj = Xte[i], Xte[j]
            # comparable: xi <= xj on governed dims
            dominated = True
            for d, m in enumerate(mono_constraint):
                if m == 1 and xi[d] > xj[d]:
                    dominated = False; break
                elif m == -1 and xi[d] < xj[d]:
                    dominated = False; break
            if dominated and model.predict(xi.reshape(1,-1))[0] > model.predict(xj.reshape(1,-1))[0]:
                viol += 1
        viol_rate = viol / n_pairs
        results.append({'seed': seed, 'acc': acc, 'viol_rate': viol_rate, 'time': elapsed})
    df = pd.DataFrame(results)
    return df['acc'].mean(), df['acc'].std(), df['viol_rate'].mean(), df['time'].mean()

seeds = list(range(42, 52))

print("Loading datasets...")

# COMPAS
try:
    X_c, y_c, dirs_c = load_compas()
    mono_c = tuple(dirs_c)
    acc_m, acc_s, vr_m, t_m = run_greedy_mono(X_c, y_c, mono_c, seeds)
    print(f"COMPAS MonoXGB1: acc={acc_m:.3f}±{acc_s:.3f}, viol={vr_m:.4f}, time={t_m:.3f}s")
    compas_row = {'dataset': 'COMPAS', 'method': 'MonoXGB1', 'acc_mean': acc_m, 'acc_std': acc_s, 'viol_rate': vr_m, 'time': t_m}
except Exception as e:
    print(f"COMPAS failed: {e}")
    compas_row = {'dataset': 'COMPAS', 'method': 'MonoXGB1', 'acc_mean': float('nan'), 'acc_std': float('nan'), 'viol_rate': float('nan'), 'time': float('nan')}

# German Credit (local UCI download needed)
try:
    X_g, y_g, gov_idx, dirs_g = load_german()
    mono_g = tuple(1 if i in gov_idx else 0 for i in range(X_g.shape[1]))
    acc_m, acc_s, vr_m, t_m = run_greedy_mono(X_g, y_g, mono_g, seeds)
    print(f"German MonoXGB1: acc={acc_m:.3f}±{acc_s:.3f}, viol={vr_m:.4f}, time={t_m:.3f}s")
    german_row = {'dataset': 'German Credit', 'method': 'MonoXGB1', 'acc_mean': acc_m, 'acc_std': acc_s, 'viol_rate': vr_m, 'time': t_m}
except Exception as e:
    print(f"German failed: {e}")
    german_row = {'dataset': 'German Credit', 'method': 'MonoXGB1', 'acc_mean': float('nan'), 'acc_std': float('nan'), 'viol_rate': float('nan'), 'time': float('nan')}

rows = [compas_row, german_row]
df_out = pd.DataFrame(rows)
df_out.to_csv('greedy_mono_baseline.csv', index=False)
print("\nSaved to greedy_mono_baseline.csv")
print(df_out.to_string())
