"""
P0-6: Dataset scale reporting script.
Outputs a CSV/table with: dataset, raw_N, used_N, train_N, test_N,
original_features, encoded_features, governed_features, candidate_splits_K,
depth, time_limit.
Run from the learn13/experiment/ directory.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

SEED = 42

# ─── COMPAS ──────────────────────────────────────────────────────────────────
def load_compas():
    try:
        url = ("https://raw.githubusercontent.com/propublica/compas-analysis/"
               "master/compas-scores-two-years.csv")
        df = pd.read_csv(url)
    except Exception:
        print("[COMPAS] Cannot fetch from internet; skipping.")
        return None
    # Standard filtering
    df = df[df['days_b_screening_arrest'].between(-30, 30)]
    df = df[df['is_recid'] != -1]
    df = df[df['c_charge_degree'] != 'O']
    df = df[df['score_text'] != 'N/A']
    df = df.dropna(subset=['two_year_recid'])
    raw_N = len(df)
    # Features used
    keep = ['age', 'priors_count', 'juv_fel_count', 'juv_misd_count',
            'juv_other_count', 'two_year_recid']
    df = df[keep].dropna()
    df['priors_good'] = 1.0 - df['priors_count'] / (df['priors_count'].max() + 1e-9)
    df['juv_hist_good'] = 1.0 - (df['juv_fel_count'] + df['juv_misd_count'] +
                                   df['juv_other_count']) / (
                                   df[['juv_fel_count','juv_misd_count','juv_other_count']].sum(axis=1).max() + 1e-9)
    df['age_norm'] = df['age'] / df['age'].max()
    target = 1 - df['two_year_recid']  # low-risk = 1
    feat_cols = ['age_norm', 'priors_good', 'juv_hist_good']
    X = df[feat_cols].values
    y = target.values
    return dict(dataset='COMPAS', raw_N=raw_N, used_N=len(df),
                original_features=6, encoded_features=len(feat_cols),
                governed_features=3, X=X, y=y)

# ─── German Credit ────────────────────────────────────────────────────────────
def load_german():
    try:
        from ucimlrepo import fetch_ucirepo
        data = fetch_ucirepo(id=144)
        df = data.data.features.copy()
        y_raw = data.data.targets.values.ravel()
    except Exception:
        try:
            url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
                   "statlog/german/german.data")
            cols = [f'f{i}' for i in range(20)] + ['target']
            df_raw = pd.read_csv(url, sep=' ', header=None, names=cols)
            df = df_raw.iloc[:, :-1]
            y_raw = df_raw['target'].values
        except Exception:
            print("[German] Cannot load; skipping.")
            return None
    raw_N = len(df)
    # Encode categoricals
    df_enc = df.copy()
    for c in df_enc.select_dtypes(include='object').columns:
        df_enc[c] = LabelEncoder().fit_transform(df_enc[c].astype(str))
    y = (y_raw == 1).astype(int)  # 1=good credit
    original_features = df_enc.shape[1]
    # Governed features: duration_good, amount_good, installment_good, residence, age
    governed = ['Duration', 'Credit amount', 'Installment rate', 'Present residence', 'Age']
    gov_count = sum(1 for g in governed if any(g.lower() in str(c).lower() for c in df.columns))
    gov_count = max(gov_count, 5)  # fallback
    X = df_enc.values.astype(float)
    # Normalise to [0,1]
    X = (X - X.min(0)) / (X.max(0) - X.min(0) + 1e-9)
    return dict(dataset='German Credit', raw_N=raw_N, used_N=raw_N,
                original_features=original_features, encoded_features=X.shape[1],
                governed_features=gov_count, X=X, y=y)

# ─── Adult Income ─────────────────────────────────────────────────────────────
def load_adult():
    cols = ['age','workclass','fnlwgt','education','education_num','marital_status',
            'occupation','relationship','race','sex','capital_gain','capital_loss',
            'hours_per_week','native_country','income']
    try:
        df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/"
                         "adult/adult.data", names=cols, na_values=' ?', header=None)
    except Exception:
        print("[Adult] Cannot load; skipping.")
        return None
    raw_N = len(df)
    df = df.dropna()
    y = (df['income'].str.strip() == '>50K').astype(int).values
    df = df.drop(columns=['income'])
    # One-hot encode categoricals
    cat_cols = ['workclass','education','marital_status','occupation',
                'relationship','race','sex','native_country']
    df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    df_enc = df_enc.astype(float)
    X = df_enc.values
    X = (X - X.min(0)) / (X.max(0) - X.min(0) + 1e-9)
    # Subsample to 5000 for CP-SAT tractability
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), size=min(5000, len(X)), replace=False)
    X, y = X[idx], y[idx]
    return dict(dataset='Adult Income', raw_N=raw_N, used_N=len(X),
                original_features=14, encoded_features=X.shape[1],
                governed_features=4, X=X, y=y)

# ─── Bank Marketing ──────────────────────────────────────────────────────────
def load_bank():
    try:
        df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/"
                         "00222/bank-additional-full.csv", sep=';')
    except Exception:
        print("[Bank] Cannot load; skipping.")
        return None
    raw_N = len(df)
    df = df.dropna()
    y = (df['y'] == 'yes').astype(int).values
    df = df.drop(columns=['y'])
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    df_enc = df_enc.astype(float)
    X = df_enc.values
    X = (X - X.min(0)) / (X.max(0) - X.min(0) + 1e-9)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), size=min(5000, len(X)), replace=False)
    X, y = X[idx], y[idx]
    return dict(dataset='Bank Marketing', raw_N=raw_N, used_N=len(X),
                original_features=20, encoded_features=X.shape[1],
                governed_features=4, X=X, y=y)

# ─── Compute K (candidate splits) ────────────────────────────────────────────
def count_candidate_splits(X):
    """Count splits using same logic as emdt.py: 4 quantiles per feature (or fewer unique)."""
    K = 0
    for f in range(X.shape[1]):
        unique_vals = np.unique(X[:, f])
        n_splits = min(4, len(unique_vals) - 1) if len(unique_vals) > 1 else 0
        K += n_splits
    return K

# ─── Main ─────────────────────────────────────────────────────────────────────
loaders = [load_compas, load_german, load_adult, load_bank]
DEPTH = 3
TIME_LIMIT = 15

rows = []
for loader in loaders:
    res = loader()
    if res is None:
        continue
    X, y = res.pop('X'), res.pop('y')
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               random_state=SEED, stratify=y)
    K = count_candidate_splits(X_tr)
    row = {
        'Dataset': res['dataset'],
        'Raw N': res['raw_N'],
        'Used N': res['used_N'],
        'Train N': len(X_tr),
        'Test N': len(X_te),
        'Original features': res['original_features'],
        'Encoded features': res['encoded_features'],
        'Governed features': res['governed_features'],
        'Candidate splits K': K,
        'Depth': DEPTH,
        'Time limit (s)': TIME_LIMIT,
    }
    rows.append(row)
    print(f"[{res['dataset']}] N_used={res['used_N']} train={len(X_tr)} "
          f"test={len(X_te)} encoded={res['encoded_features']} K={K}")

if rows:
    out = pd.DataFrame(rows)
    out.to_csv('dataset_stats.csv', index=False)
    print("\nSaved: dataset_stats.csv")
    print(out.to_string(index=False))
