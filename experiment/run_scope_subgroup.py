import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from emdt import ExactMonotonicTree
from run_revision_nmi import (
    EMDT_TIME_LIMIT,
    bootstrap_ci,
    load_compas_dataset,
    load_german_dataset,
    monotonic_violations,
    safe_predict_score,
)


SEEDS = list(range(42, 52))
FP_COST = 5.0
FN_COST = 2.0


def compute_cost(y_true, y_score):
    y_pred = (y_score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return FP_COST * fpr + FN_COST * fnr


def evaluate_model(model, X_train, y_train, X_test, y_test, n_dim, seed):
    model.fit(X_train, y_train)
    y_score = safe_predict_score(model, X_test)
    y_pred = (y_score >= 0.5).astype(int)
    viol = monotonic_violations(model, n_dim=n_dim, seed=seed + 5000)
    stats = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Cost": float(compute_cost(y_test, y_score)),
        "ViolRate": float(viol["viol_rate"]),
        "ViolSeverity": float(viol["viol_severity_mean"]),
        "Time": float(getattr(model, "fit_stats_", {}).get("wall_time", np.nan)),
        "Gap": float(getattr(model, "fit_stats_", {}).get("relative_gap", np.nan)),
    }
    return stats, y_score


def monotonic_violations_subset(model, n_dim, monotonic_idx, seed, n_checks=2000):
    rng = np.random.default_rng(seed)
    x_low = rng.random((n_checks, n_dim))
    x_high = x_low.copy()
    for j in monotonic_idx:
        a = rng.random(n_checks)
        b = rng.random(n_checks)
        x_low[:, j] = np.minimum(a, b)
        x_high[:, j] = np.maximum(a, b)

    p_low = safe_predict_score(model, x_low)
    p_high = safe_predict_score(model, x_high)
    delta = p_low - p_high
    return {
        "viol_rate": float(np.mean(delta > 1e-8)),
        "viol_severity": float(np.maximum(delta, 0.0).mean()),
    }


def summarize_table(df, group_cols, metric_cols):
    rows = []
    for keys, grp in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for m in metric_cols:
            vals = grp[m].to_numpy(dtype=float)
            row[f"{m}_Mean"] = float(np.nanmean(vals))
            row[f"{m}_Std"] = float(np.nanstd(vals, ddof=0))
            finite_vals = vals[np.isfinite(vals)]
            if len(finite_vals) >= 2:
                lo, hi = bootstrap_ci(finite_vals, seed=123)
            elif len(finite_vals) == 1:
                lo, hi = float(finite_vals[0]), float(finite_vals[0])
            else:
                lo, hi = np.nan, np.nan
            row[f"{m}_CI_L"] = lo
            row[f"{m}_CI_U"] = hi
        row["Runs"] = int(len(grp))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def _scope_seed_worker(seed, X, y, time_limit):
    records = []
    n_dim = X.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    models = {
        "ODT-none": (
            ExactMonotonicTree(
                max_depth=3,
                monotonic=False,
                time_limit=time_limit,
                num_workers=1,
                random_seed=seed,
            ),
            [],
        ),
        "EMDT-cons2": (
            ExactMonotonicTree(
                max_depth=3,
                monotonic=True,
                monotonic_features=[0, 1],
                time_limit=time_limit,
                num_workers=1,
                random_seed=seed,
            ),
            [0, 1],
        ),
        "EMDT-mid3": (
            ExactMonotonicTree(
                max_depth=3,
                monotonic=True,
                monotonic_features=[0, 1, 2],
                time_limit=time_limit,
                num_workers=1,
                random_seed=seed,
            ),
            [0, 1, 2],
        ),
        "EMDT-full5": (
            ExactMonotonicTree(
                max_depth=3,
                monotonic=True,
                monotonic_features=[0, 1, 2, 3, 4],
                time_limit=time_limit,
                num_workers=1,
                random_seed=seed,
            ),
            [0, 1, 2, 3, 4],
        ),
    }

    for name, (model, declared_idx) in models.items():
        stats, _ = evaluate_model(model, X_train, y_train, X_test, y_test, n_dim=n_dim, seed=seed)
        if declared_idx:
            declared = monotonic_violations_subset(
                model=model,
                n_dim=n_dim,
                monotonic_idx=declared_idx,
                seed=seed + 9000,
            )
            declared_rate = float(declared["viol_rate"])
            declared_sev = float(declared["viol_severity"])
        else:
            declared_rate = np.nan
            declared_sev = np.nan

        records.append(
            {
                "Seed": seed,
                "Setting": name,
                "Accuracy": stats["Accuracy"],
                "Cost": stats["Cost"],
                "ViolRate_FullPolicy": stats["ViolRate"],
                "ViolSeverity_FullPolicy": stats["ViolSeverity"],
                "ViolRate_Declared": declared_rate,
                "ViolSeverity_Declared": declared_sev,
                "Time": stats["Time"],
                "Gap": stats["Gap"],
            }
        )

    return records


def run_scope_sensitivity(seeds=None, time_limit=EMDT_TIME_LIMIT, workers=1):
    if seeds is None:
        seeds = list(SEEDS)
    data = load_german_dataset()
    X, y = data["X"], data["y"]

    records = []
    if workers <= 1:
        for seed in seeds:
            records.extend(_scope_seed_worker(seed=seed, X=X, y=y, time_limit=time_limit))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(_scope_seed_worker, seed, X, y, time_limit): seed
                for seed in seeds
            }
            for fut in as_completed(futures):
                records.extend(fut.result())
                print(f"[scope] finished seed {futures[fut]}")

    run_df = pd.DataFrame(records)
    summary_df = summarize_table(
        run_df,
        group_cols=["Setting"],
        metric_cols=[
            "Accuracy",
            "Cost",
            "ViolRate_FullPolicy",
            "ViolSeverity_FullPolicy",
            "ViolRate_Declared",
            "ViolSeverity_Declared",
            "Time",
            "Gap",
        ],
    )
    return run_df, summary_df


def subgroup_masks(dataset_name, X):
    if dataset_name == "compas_real":
        masks = {
            "age_lt_0.4": X[:, 0] < 0.4,
            "age_ge_0.4": X[:, 0] >= 0.4,
            "priors_good_ge_0.7": X[:, 1] >= 0.7,
            "priors_good_lt_0.7": X[:, 1] < 0.7,
        }
        shift_mask = X[:, 1] < 0.6
        shift_name = "shift_high_prior"
    else:
        masks = {
            "amount_good_lt_0.5": X[:, 1] < 0.5,
            "amount_good_ge_0.5": X[:, 1] >= 0.5,
            "age_lt_0.4": X[:, 4] < 0.4,
            "age_ge_0.4": X[:, 4] >= 0.4,
        }
        shift_mask = X[:, 1] < 0.5
        shift_name = "shift_high_credit_burden"
    return masks, shift_mask, shift_name


def _subgroup_seed_worker(dataset_name, X, y, seed, time_limit):
    records = []
    n_dim = X.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    models = {
        "CART": DecisionTreeClassifier(max_depth=3, random_state=seed),
        "EMDT-full": ExactMonotonicTree(
            max_depth=3,
            monotonic=True,
            monotonic_features=list(range(n_dim)),
            time_limit=time_limit,
            num_workers=1,
            random_seed=seed,
        ),
    }

    subgroup_defs, shift_mask, shift_name = subgroup_masks(dataset_name, X_test)

    for method, model in models.items():
        stats, y_score = evaluate_model(model, X_train, y_train, X_test, y_test, n_dim=n_dim, seed=seed)

        subgroup_acc = []
        subgroup_cost = []
        for gname, gmask in subgroup_defs.items():
            if np.sum(gmask) < 3:
                continue
            acc = accuracy_score(y_test[gmask], (y_score[gmask] >= 0.5).astype(int))
            cost = compute_cost(y_test[gmask], y_score[gmask])
            subgroup_acc.append(float(acc))
            subgroup_cost.append(float(cost))
            records.append(
                {
                    "Dataset": dataset_name,
                    "Seed": seed,
                    "Method": method,
                    "SliceType": "subgroup",
                    "Slice": gname,
                    "Accuracy": float(acc),
                    "Cost": float(cost),
                }
            )

        if np.sum(shift_mask) >= 3:
            shift_acc = accuracy_score(y_test[shift_mask], (y_score[shift_mask] >= 0.5).astype(int))
            shift_cost = compute_cost(y_test[shift_mask], y_score[shift_mask])
        else:
            shift_acc = np.nan
            shift_cost = np.nan

        records.append(
            {
                "Dataset": dataset_name,
                "Seed": seed,
                "Method": method,
                "SliceType": "overall_gap",
                "Slice": "overall",
                "Accuracy": stats["Accuracy"],
                "Cost": stats["Cost"],
                "WorstSubgroupAccGap": float(stats["Accuracy"] - min(subgroup_acc)) if subgroup_acc else np.nan,
                "WorstSubgroupCostGap": float(max(subgroup_cost) - stats["Cost"]) if subgroup_cost else np.nan,
                "ShiftSlice": shift_name,
                "ShiftAccDelta": float(stats["Accuracy"] - shift_acc) if np.isfinite(shift_acc) else np.nan,
                "ShiftCostDelta": float(shift_cost - stats["Cost"]) if np.isfinite(shift_cost) else np.nan,
            }
        )
    return records


def run_subgroup_stability(seeds=None, time_limit=EMDT_TIME_LIMIT, workers=1):
    if seeds is None:
        seeds = list(SEEDS)
    datasets = [load_compas_dataset(), load_german_dataset()]
    records = []

    if workers <= 1:
        for ds in datasets:
            for seed in seeds:
                records.extend(
                    _subgroup_seed_worker(
                        dataset_name=ds["name"],
                        X=ds["X"],
                        y=ds["y"],
                        seed=seed,
                        time_limit=time_limit,
                    )
                )
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {}
            for ds in datasets:
                for seed in seeds:
                    fut = ex.submit(
                        _subgroup_seed_worker,
                        ds["name"],
                        ds["X"],
                        ds["y"],
                        seed,
                        time_limit,
                    )
                    futures[fut] = (ds["name"], seed)
            for fut in as_completed(futures):
                records.extend(fut.result())
                ds_name, seed = futures[fut]
                print(f"[subgroup] finished dataset={ds_name} seed={seed}")

    raw_df = pd.DataFrame(records)
    gap_df = raw_df[raw_df["SliceType"] == "overall_gap"].copy()
    summary_df = summarize_table(
        gap_df,
        group_cols=["Dataset", "Method"],
        metric_cols=[
            "Accuracy",
            "Cost",
            "WorstSubgroupAccGap",
            "WorstSubgroupCostGap",
            "ShiftAccDelta",
            "ShiftCostDelta",
        ],
    )
    return raw_df, summary_df


def parse_int_list(text):
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main():
    warnings.filterwarnings("ignore")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    parser = argparse.ArgumentParser(description="Run scope sensitivity and subgroup stability analyses.")
    parser.add_argument(
        "--mode",
        choices=["scope", "subgroup", "both"],
        default="both",
        help="Which analysis to run.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in SEEDS),
        help="Comma-separated random seeds.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=float(EMDT_TIME_LIMIT),
        help="Time limit (seconds) for exact models.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel workers for seed-level jobs.",
    )
    args = parser.parse_args()

    seeds = parse_int_list(args.seeds)
    out_dir = Path(__file__).resolve().parent

    if args.mode in {"scope", "both"}:
        scope_runs, scope_summary = run_scope_sensitivity(
            seeds=seeds,
            time_limit=float(args.time_limit),
            workers=max(1, int(args.workers)),
        )
        scope_runs.to_csv(out_dir / "scope_sensitivity_runs.csv", index=False)
        scope_summary.to_csv(out_dir / "scope_sensitivity_summary.csv", index=False)
        print("Saved scope files.")

    if args.mode in {"subgroup", "both"}:
        subgroup_raw, subgroup_summary = run_subgroup_stability(
            seeds=seeds,
            time_limit=float(args.time_limit),
            workers=max(1, int(args.workers)),
        )
        subgroup_raw.to_csv(out_dir / "subgroup_slices.csv", index=False)
        subgroup_summary.to_csv(out_dir / "subgroup_stability_summary.csv", index=False)
        print("Saved subgroup files.")


if __name__ == "__main__":
    main()
