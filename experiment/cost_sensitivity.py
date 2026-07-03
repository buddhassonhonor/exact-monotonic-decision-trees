"""
P1-14: Cost-weight sensitivity analysis.
Computes weighted cost = w_FP * FPR + w_FN * FNR for multiple weight pairs
across all real datasets, using the already-computed revision_runs.csv.
Weight pairs: (1,1), (2,5), (5,2), (10,1).
"""
import numpy as np
import pandas as pd

# Weight pairs (w_FP, w_FN)
WEIGHT_PAIRS = [(1, 1), (2, 5), (5, 2), (10, 1)]

def weighted_cost(fpr, fnr, w_fp, w_fn):
    return w_fp * fpr + w_fn * fnr

try:
    df = pd.read_csv('revision_runs.csv')
except FileNotFoundError:
    print("ERROR: revision_runs.csv not found.")
    raise

print("Columns:", df.columns.tolist())

# Detect FPR/FNR column names (handles uppercase or lowercase)
if 'FPR' in df.columns and 'FNR' in df.columns:
    fpr_col, fnr_col = 'FPR', 'FNR'
elif 'fpr' in df.columns and 'fnr' in df.columns:
    fpr_col, fnr_col = 'fpr', 'fnr'
else:
    print("FPR/FNR columns not found. Available:", df.columns.tolist())
    raise SystemExit(1)

# Normalise column names for groupby
df = df.rename(columns={'Dataset': 'dataset', 'Method': 'method',
                         fpr_col: 'fpr', fnr_col: 'fnr'})


# ─── If run-level fpr/fnr available ──────────────────────────────────────────
if 'fpr' in df.columns and 'fnr' in df.columns:
    rows = []
    for (wp, wn) in WEIGHT_PAIRS:
        tmp = df.copy()
        tmp['cost'] = weighted_cost(tmp['fpr'], tmp['fnr'], wp, wn)
        grp = (tmp.groupby(['dataset', 'method'])['cost']
               .agg(['mean', 'std']).reset_index())
        grp.columns = ['dataset', 'method', f'cost_{wp}_{wn}_mean', f'cost_{wp}_{wn}_std']
        rows.append(grp)
    out = rows[0]
    for r in rows[1:]:
        out = out.merge(r, on=['dataset', 'method'])
    out.to_csv('cost_sensitivity.csv', index=False)
    print("Saved: cost_sensitivity.csv")
    print(out.to_string(index=False))
else:
    # Fallback: report what columns we have
    print("FPR/FNR columns not found in revision_runs.csv.")
    print("Available:", df.columns.tolist())
    print("Please verify the experiment output file structure.")

# ─── Pivot table for paper ────────────────────────────────────────────────────
if 'fpr' in df.columns and 'fnr' in df.columns:
    print("\n" + "=" * 60)
    print("Cost sensitivity pivot (mean over seeds):")
    print("=" * 60)
    for ds in out['dataset'].unique():
        sub = out[out['dataset'] == ds]
        print(f"\n{ds}:")
        for _, row in sub.iterrows():
            vals = " | ".join(
                f"({wp},{wn}): {row[f'cost_{wp}_{wn}_mean']:.3f}±{row[f'cost_{wp}_{wn}_std']:.3f}"
                for (wp, wn) in WEIGHT_PAIRS
            )
            print(f"  {row['method']:15s}  {vals}")
