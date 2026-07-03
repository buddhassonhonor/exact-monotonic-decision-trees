"""
P1-12 (batch): Depth ablation 10 seeds — runs one depth at a time.
Usage: python rerun_ablation_bydepth.py --depth 2
       python rerun_ablation_bydepth.py --depth 3
       python rerun_ablation_bydepth.py --depth 4
Or:    python rerun_ablation_bydepth.py   (runs all depths sequentially, ~3 min)
"""
import argparse, numpy as np, pandas as pd, time, os
from emdt import ExactMonotonicTree
from sklearn.model_selection import train_test_split

SEEDS   = list(range(42, 52))
TLIMIT  = 15
OUT     = 'depth_ablation_10seeds.csv'
SUMOUT  = 'depth_ablation_10seeds_summary.csv'

def make_credit(n=150, noise=0.15, seed=42):
    rng = np.random.default_rng(seed)
    X   = rng.uniform(0, 1, (n, 3))
    y   = (0.5*X[:,0]+0.3*X[:,1]+0.2*X[:,2] > 0.5).astype(int)
    y[rng.uniform(size=n) < noise] ^= 1
    return X, y

parser = argparse.ArgumentParser()
parser.add_argument('--depth', type=int, default=0,
                    help='Depth to run (2/3/4). 0=all.')
args = parser.parse_args()
depths = [args.depth] if args.depth in [2,3,4] else [2,3,4]

# Load existing partial results
existing = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame()
done_keys = set(zip(existing.get('depth',[]), existing.get('seed',[]))) \
            if not existing.empty else set()

rows = existing.to_dict('records') if not existing.empty else []

for depth in depths:
    for seed in SEEDS:
        if (depth, seed) in done_keys:
            print(f"  depth={depth} seed={seed} — skip (already done)")
            continue
        X, y = make_credit(seed=seed)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y)
        t0  = time.time()
        clf = ExactMonotonicTree(max_depth=depth, time_limit=TLIMIT,
                                 monotonic=True, random_seed=seed)
        ok  = clf.fit(Xtr, ytr)
        el  = time.time() - t0
        tra = float((clf.predict(Xtr)==ytr).mean()) if ok else float('nan')
        tea = float((clf.predict(Xte)==yte).mean()) if ok else float('nan')
        st  = clf.fit_stats_.get('status','N/A')
        gap = clf.fit_stats_.get('relative_gap', float('nan'))
        rows.append(dict(depth=depth, seed=seed, time=el,
                         train_acc=tra, test_acc=tea, status=st, gap=gap))
        pd.DataFrame(rows).to_csv(OUT, index=False)   # checkpoint
        print(f"  depth={depth} seed={seed} t={el:.2f}s "
              f"tr={tra:.3f} te={tea:.3f} [{st}]", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)

if len(df) >= 30:   # all 10 seeds × 3 depths
    s = (df.groupby('depth')
         .agg(time_mean=('time','mean'), time_std=('time','std'),
              train_acc_mean=('train_acc','mean'), train_acc_std=('train_acc','std'),
              test_acc_mean=('test_acc','mean'),  test_acc_std=('test_acc','std'))
         .reset_index())
    s.to_csv(SUMOUT, index=False)
    print("\n=== SUMMARY ===")
    print(s.to_string(index=False))
else:
    print(f"\nPartial: {len(df)}/30 runs saved to {OUT}")
