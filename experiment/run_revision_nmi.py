import argparse
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.optimize import minimize
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from emdt import ExactMonotonicTree


SEEDS = list(range(42, 52))
TEST_SIZE = 0.2
N_VIOLATION_CHECKS = 2000
BOOTSTRAP_REPEATS = 2000
EMDT_TIME_LIMIT = 15
ANYTIME_BUDGETS = [1, 3, 5, 10, 15]
MITIGATION_WORKERS = [1, 4]
MITIGATION_DATASET = "german_real"
MITIGATION_BUDGET = 15
MITIGATION_INIT_BUDGET = 3
FP_COST = 5.0
FN_COST = 2.0
MAX_SAMPLES_PER_DATASET = 500


def to_unit_interval(values, invert=False):
    arr = np.asarray(values, dtype=float)
    lo = np.nanmin(arr)
    hi = np.nanmax(arr)
    if hi - lo < 1e-12:
        scaled = np.zeros_like(arr)
    else:
        scaled = (arr - lo) / (hi - lo)
    if invert:
        scaled = 1.0 - scaled
    return np.clip(scaled, 0.0, 1.0)


def load_compas_dataset():
    path = Path("D:/data/compas/compas-scores-two-years.csv")
    df = pd.read_csv(path)

    # Standard filtering protocol used in prior COMPAS audits.
    df = df[
        (df["days_b_screening_arrest"] <= 30)
        & (df["days_b_screening_arrest"] >= -30)
        & (df["is_recid"] != -1)
        & (df["c_charge_degree"] != "O")
        & (df["score_text"] != "N/A")
    ].copy()

    cols = ["age", "priors_count", "juv_fel_count", "juv_misd_count", "juv_other_count", "two_year_recid"]
    df = df[cols].dropna()
    if len(df) > MAX_SAMPLES_PER_DATASET:
        df = df.sample(n=MAX_SAMPLES_PER_DATASET, random_state=2026, replace=False).reset_index(drop=True)

    # Output label is "low risk" (1) so that increasing features should not lower approval probability.
    y = 1 - df["two_year_recid"].astype(int).to_numpy()
    X = np.column_stack(
        [
            to_unit_interval(df["age"], invert=False),
            to_unit_interval(df["priors_count"], invert=True),
            to_unit_interval(df["juv_fel_count"], invert=True),
            to_unit_interval(df["juv_misd_count"], invert=True),
            to_unit_interval(df["juv_other_count"], invert=True),
        ]
    )
    feature_names = [
        "age",
        "priors_good",
        "juv_fel_good",
        "juv_misd_good",
        "juv_other_good",
    ]
    return {
        "name": "compas_real",
        "X": X,
        "y": y,
        "feature_names": feature_names,
    }


def load_german_dataset():
    path = Path("D:/data/german/german.data")
    df = pd.read_csv(path, sep=r"\s+", header=None)

    if len(df) > MAX_SAMPLES_PER_DATASET:
        df = df.sample(n=MAX_SAMPLES_PER_DATASET, random_state=2026, replace=False).reset_index(drop=True)

    # Numeric features selected for monotonic governance assumptions.
    duration = to_unit_interval(df[1], invert=True)  # Shorter horizon is safer.
    amount = to_unit_interval(df[4], invert=True)  # Smaller credit amount is safer.
    installment = to_unit_interval(df[7], invert=True)  # Lower installment burden is safer.
    residence = to_unit_interval(df[10], invert=False)  # Longer residence often indicates stability.
    age = to_unit_interval(df[12], invert=False)  # Older applicants are often lower-risk in scorecards.

    X = np.column_stack([duration, amount, installment, residence, age])
    y = (df[20].astype(int) == 1).astype(int).to_numpy()  # 1 means good credit in original coding.
    feature_names = ["duration_good", "amount_good", "installment_good", "residence", "age"]
    return {
        "name": "german_real",
        "X": X,
        "y": y,
        "feature_names": feature_names,
    }


def load_adult_dataset():
    path = Path("D:/data/adult/adult.data")
    cols = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        "income",
    ]
    df = pd.read_csv(path, names=cols, sep=r",\s*", engine="python", na_values=["?"])
    df = df[
        ["age", "education_num", "capital_gain", "hours_per_week", "income"]
    ].dropna()
    if len(df) > MAX_SAMPLES_PER_DATASET:
        df = df.sample(n=MAX_SAMPLES_PER_DATASET, random_state=2026, replace=False).reset_index(drop=True)

    y = (df["income"].astype(str).str.strip() == ">50K").astype(int).to_numpy()
    X = np.column_stack(
        [
            to_unit_interval(df["age"], invert=False),
            to_unit_interval(df["education_num"], invert=False),
            to_unit_interval(df["capital_gain"], invert=False),
            to_unit_interval(df["hours_per_week"], invert=False),
        ]
    )
    feature_names = ["age", "education_num", "capital_gain", "hours_per_week"]
    return {
        "name": "adult_income",
        "X": X,
        "y": y,
        "feature_names": feature_names,
    }


def load_bank_dataset():
    path = Path("D:/data/bank/bank-full.csv")
    df = pd.read_csv(path, sep=";")
    df = df[["balance", "duration", "campaign", "previous", "y"]].dropna()
    if len(df) > MAX_SAMPLES_PER_DATASET:
        df = df.sample(n=MAX_SAMPLES_PER_DATASET, random_state=2026, replace=False).reset_index(drop=True)

    y = (df["y"].astype(str).str.strip().str.lower() == "yes").astype(int).to_numpy()
    X = np.column_stack(
        [
            to_unit_interval(df["balance"], invert=False),
            to_unit_interval(df["duration"], invert=False),
            to_unit_interval(df["campaign"], invert=True),  # fewer contacts usually indicate easier conversion
            to_unit_interval(df["previous"], invert=False),
        ]
    )
    feature_names = ["balance", "duration", "campaign_good", "previous"]
    return {
        "name": "bank_marketing",
        "X": X,
        "y": y,
        "feature_names": feature_names,
    }


class NonNegativeLogistic:
    def __init__(self, l2=1e-3, max_iter=400):
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.intercept_ = 0.0
        self.coef_ = None
        self.success_ = False

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape

        def objective(theta):
            b = theta[0]
            w = theta[1:]
            z = b + X @ w
            p = self._sigmoid(z)
            eps = 1e-12
            loss = -np.mean(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))
            loss += 0.5 * self.l2 * np.dot(w, w)
            grad_common = p - y
            grad_b = np.mean(grad_common)
            grad_w = (X.T @ grad_common) / n_samples + self.l2 * w
            grad = np.concatenate(([grad_b], grad_w))
            return loss, grad

        x0 = np.zeros(n_features + 1, dtype=float)
        bounds = [(None, None)] + [(0.0, None)] * n_features
        result = minimize(
            fun=lambda t: objective(t)[0],
            x0=x0,
            jac=lambda t: objective(t)[1],
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.max_iter},
        )
        self.success_ = bool(result.success)
        self.intercept_ = float(result.x[0])
        self.coef_ = result.x[1:].astype(float)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        z = self.intercept_ + X @ self.coef_
        p1 = self._sigmoid(z)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X):
        p = self.predict_proba(X)[:, 1]
        return (p >= 0.5).astype(int)


class MonotoneCalibratedModel:
    """Post-hoc isotonic calibration wrapper that preserves score monotonic order."""

    def __init__(self, base_model):
        self.base_model = base_model
        self.calibrator_ = None
        self.constant_prob_ = None
        self.fit_stats_ = {}

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.base_model.fit(X, y.astype(int))
        train_score = safe_predict_score(self.base_model, X)
        if np.allclose(train_score, train_score[0]):
            self.constant_prob_ = float(np.mean(y))
            self.calibrator_ = None
        else:
            self.calibrator_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self.calibrator_.fit(train_score, y)
            self.constant_prob_ = None

        base_stats = getattr(self.base_model, "fit_stats_", {})
        self.fit_stats_ = dict(base_stats)
        self.fit_stats_["post_calibration"] = "isotonic"
        return self

    def predict_proba(self, X):
        base_score = safe_predict_score(self.base_model, X)
        if self.calibrator_ is None:
            p1 = np.full(base_score.shape, float(self.constant_prob_), dtype=float)
        else:
            p1 = self.calibrator_.transform(base_score)
        p1 = np.clip(np.asarray(p1, dtype=float), 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        p = self.predict_proba(X)[:, 1]
        return (p >= 0.5).astype(int)


def safe_predict_score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    return model.predict(X).astype(float)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not np.any(mask):
            continue
        acc = np.mean(y_true[mask])
        conf = np.mean(y_prob[mask])
        ece += np.mean(mask) * abs(acc - conf)
    return float(ece)


def monotonic_violations(model, n_dim, seed):
    rng = np.random.default_rng(seed)
    x1 = rng.random((N_VIOLATION_CHECKS, n_dim))
    x2 = rng.random((N_VIOLATION_CHECKS, n_dim))
    x_min = np.minimum(x1, x2)
    x_max = np.maximum(x1, x2)

    p_min = safe_predict_score(model, x_min)
    p_max = safe_predict_score(model, x_max)
    delta = p_min - p_max
    mask = delta > 1e-8
    if np.any(mask):
        cond_mean = float(delta[mask].mean())
        max_delta = float(delta[mask].max())
    else:
        cond_mean = 0.0
        max_delta = 0.0
    return {
        "violations": int(mask.sum()),
        "viol_rate": float(mask.mean()),
        "viol_severity_mean": float(np.maximum(delta, 0.0).mean()),
        "viol_severity_cond": cond_mean,
        "viol_severity_max": max_delta,
    }


def bootstrap_ci(values, repeats=BOOTSTRAP_REPEATS, alpha=0.05, seed=0):
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repeats):
        sampled = rng.choice(arr, size=len(arr), replace=True)
        samples.append(np.mean(sampled))
    lo = float(np.quantile(samples, alpha / 2.0))
    hi = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return lo, hi


def build_models(n_features, seed, emdt_time_limit=EMDT_TIME_LIMIT, include_best_practice=True):
    monotone_xgb = "(" + ",".join(["1"] * n_features) + ")"
    monotone_lgb = [1] * n_features

    models = {
        "CART": DecisionTreeClassifier(max_depth=3, random_state=seed),
        "ODT": ExactMonotonicTree(
            max_depth=3,
            monotonic=False,
            time_limit=emdt_time_limit,
            num_workers=1,
            random_seed=seed,
        ),
        "EMDT": ExactMonotonicTree(
            max_depth=3,
            monotonic=True,
            time_limit=emdt_time_limit,
            num_workers=1,
            random_seed=seed,
        ),
        "EMDT-Iso": MonotoneCalibratedModel(
            ExactMonotonicTree(
                max_depth=3,
                monotonic=True,
                time_limit=emdt_time_limit,
                num_workers=1,
                random_seed=seed,
            )
        ),
        "MonoRF": RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            monotonic_cst=monotone_lgb,
            random_state=seed,
            n_jobs=-1,
        ),
        "XGB-Mono": XGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            monotone_constraints=monotone_xgb,
            tree_method="hist",
            random_state=seed,
            n_jobs=4,
        ),
        "LGBM-Mono": LGBMClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary",
            monotone_constraints=monotone_lgb,
            random_state=seed,
            n_jobs=4,
            verbosity=-1,
        ),
        "MonoScore": NonNegativeLogistic(l2=1e-3, max_iter=500),
    }
    if include_best_practice:
        models.update(
            {
                "XGB-Mono-Best": XGBClassifier(
                    n_estimators=600,
                    max_depth=4,
                    learning_rate=0.03,
                    min_child_weight=1.0,
                    subsample=1.0,
                    colsample_bytree=1.0,
                    reg_lambda=1.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    monotone_constraints=monotone_xgb,
                    tree_method="hist",
                    random_state=seed,
                    n_jobs=4,
                ),
                "LGBM-Mono-Best": LGBMClassifier(
                    n_estimators=600,
                    max_depth=-1,
                    num_leaves=31,
                    min_child_samples=20,
                    learning_rate=0.03,
                    subsample=1.0,
                    colsample_bytree=1.0,
                    objective="binary",
                    monotone_constraints=monotone_lgb,
                    random_state=seed,
                    n_jobs=4,
                    verbosity=-1,
                ),
            }
        )
    return models


def evaluate_model(model, X_train, y_train, X_test, y_test, n_dim, seed):
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0

    y_prob = safe_predict_score(model, X_test)
    y_pred = (y_prob >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    cost = FP_COST * fpr + FN_COST * fnr

    viol = monotonic_violations(model, n_dim=n_dim, seed=seed + 1000)

    row = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Brier": float(brier_score_loss(y_test, y_prob)),
        "ECE": expected_calibration_error(y_test, y_prob, n_bins=10),
        "FPR": float(fpr),
        "FNR": float(fnr),
        "Cost": float(cost),
        "Time": float(fit_time),
        **viol,
    }

    if hasattr(model, "fit_stats_"):
        stats = model.fit_stats_
        row["SolverStatus"] = stats.get("status", "NA")
        row["Objective"] = stats.get("objective", np.nan)
        row["BestBound"] = stats.get("best_bound", np.nan)
        row["RelativeGap"] = stats.get("relative_gap", np.nan)
        row["SolverWallTime"] = stats.get("solver_wall_time", np.nan)
    else:
        row["SolverStatus"] = "NA"
        row["Objective"] = np.nan
        row["BestBound"] = np.nan
        row["RelativeGap"] = np.nan
        row["SolverWallTime"] = np.nan
    return row


def summarize_metrics(run_df, group_cols, metric_cols):
    summary_rows = []
    for keys, grp in run_df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["Runs"] = int(len(grp))
        for col in metric_cols:
            values = grp[col].to_numpy(dtype=float)
            row[f"{col}_Mean"] = float(np.nanmean(values))
            row[f"{col}_Std"] = float(np.nanstd(values, ddof=0))
            finite_values = values[np.isfinite(values)]
            if len(finite_values) >= 2:
                lo, hi = bootstrap_ci(finite_values, seed=123)
            elif len(finite_values) == 1:
                lo, hi = float(finite_values[0]), float(finite_values[0])
            else:
                lo, hi = np.nan, np.nan
            row[f"{col}_CI_L"] = lo
            row[f"{col}_CI_U"] = hi

        statuses = grp["SolverStatus"].astype(str) if "SolverStatus" in grp else pd.Series([], dtype=str)
        row["OptimalRate"] = float(np.mean(statuses == "OPTIMAL")) if len(statuses) else np.nan
        row["FeasibleRate"] = (
            float(np.mean((statuses == "OPTIMAL") | (statuses == "FEASIBLE"))) if len(statuses) else np.nan
        )
        summary_rows.append(row)
    return pd.DataFrame(summary_rows).sort_values(group_cols).reset_index(drop=True)


def run_paired_tests(run_df):
    test_rows = []
    for dataset_name, dgrp in run_df.groupby("Dataset"):
        emdt_grp = dgrp[dgrp["Method"] == "EMDT"].set_index("Seed")
        for method in sorted(dgrp["Method"].unique()):
            if method == "EMDT":
                continue
            other_grp = dgrp[dgrp["Method"] == method].set_index("Seed")
            common = emdt_grp.index.intersection(other_grp.index)
            if len(common) < 3:
                continue
            acc_e = emdt_grp.loc[common, "Accuracy"].to_numpy()
            acc_o = other_grp.loc[common, "Accuracy"].to_numpy()
            cost_e = emdt_grp.loc[common, "Cost"].to_numpy()
            cost_o = other_grp.loc[common, "Cost"].to_numpy()
            try:
                p_acc = float(wilcoxon(acc_e, acc_o, zero_method="pratt").pvalue)
            except ValueError:
                p_acc = np.nan
            try:
                p_cost = float(wilcoxon(cost_e, cost_o, zero_method="pratt").pvalue)
            except ValueError:
                p_cost = np.nan
            test_rows.append(
                {
                    "Dataset": dataset_name,
                    "Baseline": method,
                    "N": len(common),
                    "EMDT_Acc_Mean": float(np.mean(acc_e)),
                    "Base_Acc_Mean": float(np.mean(acc_o)),
                    "Acc_pvalue": p_acc,
                    "EMDT_Cost_Mean": float(np.mean(cost_e)),
                    "Base_Cost_Mean": float(np.mean(cost_o)),
                    "Cost_pvalue": p_cost,
                }
            )
    if not test_rows:
        return pd.DataFrame(
            columns=[
                "Dataset",
                "Baseline",
                "N",
                "EMDT_Acc_Mean",
                "Base_Acc_Mean",
                "Acc_pvalue",
                "EMDT_Cost_Mean",
                "Base_Cost_Mean",
                "Cost_pvalue",
            ]
        )
    return pd.DataFrame(test_rows).sort_values(["Dataset", "Baseline"]).reset_index(drop=True)


def run_revision_experiments(
    seeds=None,
    emdt_time_limit=EMDT_TIME_LIMIT,
    include_best_practice=True,
    dataset_names=None,
):
    warnings.filterwarnings("ignore")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    if seeds is None:
        seeds = list(SEEDS)

    out_dir = Path(__file__).resolve().parent
    dataset_map = {
        "compas_real": load_compas_dataset(),
        "german_real": load_german_dataset(),
        "adult_income": load_adult_dataset(),
        "bank_marketing": load_bank_dataset(),
    }
    if dataset_names is None:
        dataset_names = tuple(dataset_map.keys())
    datasets = [dataset_map[name] for name in dataset_names if name in dataset_map]
    if not datasets:
        raise ValueError(f"No valid datasets found in {dataset_names}.")

    records = []
    for data in datasets:
        X = data["X"]
        y = data["y"]
        dataset_name = data["name"]
        n_features = X.shape[1]
        print(f"Running dataset: {dataset_name} (N={len(X)}, d={n_features})")

        for seed in seeds:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=TEST_SIZE,
                random_state=seed,
                stratify=y,
            )

            models = build_models(
                n_features=n_features,
                seed=seed,
                emdt_time_limit=emdt_time_limit,
                include_best_practice=include_best_practice,
            )
            for method, model in models.items():
                print(f"  seed={seed} method={method}")
                row = evaluate_model(
                    model=model,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    n_dim=n_features,
                    seed=seed,
                )
                row["Dataset"] = dataset_name
                row["Seed"] = seed
                row["Method"] = method
                records.append(row)

    run_df = pd.DataFrame(records)
    run_path = out_dir / "revision_runs.csv"
    run_df.to_csv(run_path, index=False)
    print(f"Saved per-run metrics: {run_path}")

    metric_cols = [
        "Accuracy",
        "Brier",
        "ECE",
        "FPR",
        "FNR",
        "Cost",
        "violations",
        "viol_rate",
        "viol_severity_mean",
        "viol_severity_cond",
        "viol_severity_max",
        "Time",
        "RelativeGap",
    ]
    summary_df = summarize_metrics(
        run_df=run_df,
        group_cols=["Dataset", "Method"],
        metric_cols=metric_cols,
    )
    summary_path = out_dir / "revision_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    test_df = run_paired_tests(run_df=run_df)
    test_path = out_dir / "revision_paired_tests.csv"
    test_df.to_csv(test_path, index=False)
    print(f"Saved paired tests: {test_path}")
    return run_df, summary_df, test_df


def run_anytime_exact(
    seeds=None,
    budgets=None,
    dataset_names=("german_real",),
):
    warnings.filterwarnings("ignore")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    if seeds is None:
        seeds = list(SEEDS)
    if budgets is None:
        budgets = list(ANYTIME_BUDGETS)

    out_dir = Path(__file__).resolve().parent
    dataset_map = {
        "compas_real": load_compas_dataset(),
        "german_real": load_german_dataset(),
    }
    selected = [dataset_map[name] for name in dataset_names if name in dataset_map]
    if not selected:
        raise ValueError(f"No valid dataset names found in {dataset_names}.")

    records = []
    for data in selected:
        X = data["X"]
        y = data["y"]
        dataset_name = data["name"]
        n_features = X.shape[1]
        print(f"Running anytime dataset: {dataset_name} (N={len(X)}, d={n_features})")

        for seed in seeds:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=TEST_SIZE,
                random_state=seed,
                stratify=y,
            )
            for budget in budgets:
                for method, monotonic in [("ODT", False), ("EMDT", True)]:
                    print(f"  seed={seed} budget={budget}s method={method}")
                    model = ExactMonotonicTree(
                        max_depth=3,
                        monotonic=monotonic,
                        time_limit=float(budget),
                        num_workers=1,
                        random_seed=seed,
                    )
                    row = evaluate_model(
                        model=model,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        n_dim=n_features,
                        seed=seed,
                    )
                    row["Dataset"] = dataset_name
                    row["Seed"] = seed
                    row["Method"] = method
                    row["BudgetSec"] = float(budget)
                    records.append(row)

    run_df = pd.DataFrame(records)
    run_path = out_dir / "revision_anytime_runs.csv"
    run_df.to_csv(run_path, index=False)
    print(f"Saved anytime runs: {run_path}")

    metric_cols = [
        "Accuracy",
        "Cost",
        "viol_rate",
        "viol_severity_mean",
        "Time",
        "RelativeGap",
    ]
    summary_df = summarize_metrics(
        run_df=run_df,
        group_cols=["Dataset", "Method", "BudgetSec"],
        metric_cols=metric_cols,
    )
    summary_path = out_dir / "revision_anytime_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved anytime summary: {summary_path}")
    return run_df, summary_df


def run_solver_mitigation(
    seeds=None,
    budget=MITIGATION_BUDGET,
    dataset_name=MITIGATION_DATASET,
    workers=None,
    include_warm_start=True,
    init_budget=MITIGATION_INIT_BUDGET,
):
    warnings.filterwarnings("ignore")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    if seeds is None:
        seeds = list(SEEDS)
    if workers is None:
        workers = list(MITIGATION_WORKERS)

    workers = sorted({max(1, int(w)) for w in workers})
    out_dir = Path(__file__).resolve().parent
    dataset_map = {
        "compas_real": load_compas_dataset(),
        "german_real": load_german_dataset(),
        "adult_income": load_adult_dataset(),
        "bank_marketing": load_bank_dataset(),
    }
    if dataset_name not in dataset_map:
        raise ValueError(
            f"Unknown mitigation dataset: {dataset_name}. "
            f"Available: {sorted(dataset_map.keys())}"
        )

    data = dataset_map[dataset_name]
    X = data["X"]
    y = data["y"]
    n_features = X.shape[1]
    records = []
    print(
        f"Running solver mitigation on {dataset_name} "
        f"(N={len(X)}, d={n_features}, budget={float(budget):.1f}s, workers={workers}, warm={bool(include_warm_start)})"
    )

    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=seed,
            stratify=y,
        )

        warm_solution = None
        warm_status = "NA"
        warm_gap = np.nan
        warm_time = np.nan
        if include_warm_start:
            initializer = ExactMonotonicTree(
                max_depth=3,
                monotonic=False,
                time_limit=float(init_budget),
                num_workers=1,
                random_seed=seed,
            )
            initializer.fit(X_train, y_train)
            warm_solution = getattr(initializer, "solution_", None)
            init_stats = getattr(initializer, "fit_stats_", {})
            warm_status = str(init_stats.get("status", "NA"))
            warm_gap = float(init_stats.get("relative_gap", np.nan))
            warm_time = float(init_stats.get("solver_wall_time", np.nan))

        configs = []
        for num_workers in workers:
            configs.append(
                {
                    "Method": "ODT",
                    "Monotonic": False,
                    "Workers": int(num_workers),
                    "WarmStart": 0,
                }
            )
            configs.append(
                {
                    "Method": "EMDT",
                    "Monotonic": True,
                    "Workers": int(num_workers),
                    "WarmStart": 0,
                }
            )
        if include_warm_start and 4 in workers:
            configs.append(
                {
                    "Method": "EMDT",
                    "Monotonic": True,
                    "Workers": 4,
                    "WarmStart": 1,
                }
            )
        elif include_warm_start:
            configs.append(
                {
                    "Method": "EMDT",
                    "Monotonic": True,
                    "Workers": int(max(workers)),
                    "WarmStart": 1,
                }
            )

        for cfg in configs:
            method = cfg["Method"]
            monotonic = bool(cfg["Monotonic"])
            num_workers = int(cfg["Workers"])
            warm_flag = int(cfg["WarmStart"])
            print(f"  seed={seed} workers={num_workers} method={method} warm={warm_flag}")
            warm_hint = warm_solution if warm_flag == 1 else None
            if warm_flag == 1 and not isinstance(warm_hint, dict):
                continue
            model = ExactMonotonicTree(
                max_depth=3,
                monotonic=monotonic,
                time_limit=float(budget),
                num_workers=num_workers,
                random_seed=seed,
                warm_start_solution=warm_hint,
            )
            row = evaluate_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                n_dim=n_features,
                seed=seed,
            )
            row["Dataset"] = dataset_name
            row["Seed"] = seed
            row["Method"] = method
            row["BudgetSec"] = float(budget)
            row["Workers"] = int(num_workers)
            row["WarmStart"] = int(warm_flag)
            row["InitBudgetSec"] = float(init_budget) if warm_flag == 1 else 0.0
            row["InitStatus"] = warm_status if warm_flag == 1 else "NA"
            row["InitGap"] = warm_gap if warm_flag == 1 else np.nan
            row["InitWallTime"] = warm_time if warm_flag == 1 else np.nan
            records.append(row)

    run_df = pd.DataFrame(records)
    run_path = out_dir / "revision_solver_mitigation_runs.csv"
    run_df.to_csv(run_path, index=False)
    print(f"Saved solver mitigation runs: {run_path}")

    metric_cols = [
        "Accuracy",
        "Cost",
        "viol_rate",
        "viol_severity_mean",
        "Time",
        "SolverWallTime",
        "RelativeGap",
    ]
    summary_df = summarize_metrics(
        run_df=run_df,
        group_cols=["Dataset", "Method", "Workers", "BudgetSec", "WarmStart", "InitBudgetSec"],
        metric_cols=metric_cols,
    )
    summary_path = out_dir / "revision_solver_mitigation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved solver mitigation summary: {summary_path}")
    return run_df, summary_df


def parse_int_list(text):
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run revision experiments for EMDT paper.")
    parser.add_argument(
        "--mode",
        choices=["full", "anytime", "both", "mitigation", "all"],
        default="both",
        help="Which experiments to run.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in SEEDS),
        help="Comma-separated integer seeds.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=float(EMDT_TIME_LIMIT),
        help="Time limit (seconds) for exact models in full mode.",
    )
    parser.add_argument(
        "--no-best-practice",
        action="store_true",
        help="Disable best-practice boosting track.",
    )
    parser.add_argument(
        "--anytime-budgets",
        type=str,
        default=",".join(str(b) for b in ANYTIME_BUDGETS),
        help="Comma-separated time budgets (seconds) for anytime mode.",
    )
    parser.add_argument(
        "--anytime-datasets",
        type=str,
        default="german_real",
        help="Comma-separated dataset names for anytime mode (compas_real,german_real).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="compas_real,german_real,adult_income,bank_marketing",
        help="Comma-separated datasets for full mode.",
    )
    parser.add_argument(
        "--mitigation-dataset",
        type=str,
        default=MITIGATION_DATASET,
        help="Dataset for solver mitigation mode.",
    )
    parser.add_argument(
        "--mitigation-budget",
        type=float,
        default=float(MITIGATION_BUDGET),
        help="Time budget (seconds) for solver mitigation mode.",
    )
    parser.add_argument(
        "--mitigation-workers",
        type=str,
        default=",".join(str(w) for w in MITIGATION_WORKERS),
        help="Comma-separated worker counts for solver mitigation mode.",
    )
    parser.add_argument(
        "--mitigation-no-warm-start",
        action="store_true",
        help="Disable warm-start mitigation in mitigation mode.",
    )
    parser.add_argument(
        "--mitigation-init-budget",
        type=float,
        default=float(MITIGATION_INIT_BUDGET),
        help="Initializer time budget (seconds) for warm-start mitigation.",
    )
    args = parser.parse_args()

    seeds = parse_int_list(args.seeds)
    anytime_budgets = parse_int_list(args.anytime_budgets)
    anytime_datasets = [x.strip() for x in args.anytime_datasets.split(",") if x.strip()]
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    mitigation_workers = parse_int_list(args.mitigation_workers)
    if not mitigation_workers:
        mitigation_workers = list(MITIGATION_WORKERS)
    include_best_practice = not args.no_best_practice

    if args.mode in {"full", "both", "all"}:
        run_revision_experiments(
            seeds=seeds,
            emdt_time_limit=float(args.time_limit),
            include_best_practice=include_best_practice,
            dataset_names=tuple(datasets),
        )
    if args.mode in {"anytime", "both", "all"}:
        run_anytime_exact(
            seeds=seeds,
            budgets=anytime_budgets,
            dataset_names=tuple(anytime_datasets),
        )
    if args.mode in {"mitigation", "all"}:
        run_solver_mitigation(
            seeds=seeds,
            budget=float(args.mitigation_budget),
            dataset_name=args.mitigation_dataset.strip(),
            workers=mitigation_workers,
            include_warm_start=not args.mitigation_no_warm_start,
            init_budget=float(args.mitigation_init_budget),
        )


if __name__ == "__main__":
    main()
