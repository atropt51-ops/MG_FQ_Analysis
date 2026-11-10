"""
31_imbalance_comparison.py

Compare three strategies for handling class imbalance:
  A) Baseline (no resampling, no class weighting)
  B) Class weighting via XGBoost scale_pos_weight
  C) SMOTE inside training fold

All three use the SAME nested CV pipeline (StratifiedKFold outer,
SelectKBest inside each training fold, hyperparameter tuning inside
an inner fold) to keep results directly comparable.

Addresses reviewer concern:
  "Compare model performance in at least three settings:
   without SMOTE, with class weighting, and with SMOTE."
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, average_precision_score,
    balanced_accuracy_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

RANDOM_STATE = 42
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3
K_BEST = 100
SMOTE_K_NEIGHBORS = 5

MATRIX_FILE = "data/processed/snp_matrix_binary.tsv"
PHENO_FILE = "data/processed/phenotype_clean.tsv"
OUT_DIR = "results/imbalance_comparison"
os.makedirs(OUT_DIR, exist_ok=True)

ACCESSION_COL = "SRR/ERS_Accession "
PHENO_COL = "fluoroquinolone_phenotype"
RESISTANT_VALUE = "R"


def load_data():
    print(f"[INFO] Loading SNP matrix: {MATRIX_FILE}")
    mat = pd.read_csv(MATRIX_FILE, sep="\t")
    print(f"[INFO] Matrix shape: {mat.shape}")

    print(f"[INFO] Loading phenotype: {PHENO_FILE}")
    pheno = pd.read_csv(PHENO_FILE, sep="\t")
    pheno[ACCESSION_COL] = pheno[ACCESSION_COL].astype(str).str.strip()

    sample_cols = list(mat.columns[4:])
    pheno = pheno.set_index(ACCESSION_COL).reindex(sample_cols)
    y = (pheno[PHENO_COL] == RESISTANT_VALUE).astype(int).values
    print(f"[INFO] Class balance: R={int(y.sum())}, S={int((y==0).sum())}")

    if mat.columns[0].upper() in ("CHROM", "CHR", "#CHROM"):
        feature_names = (
            mat[mat.columns[0]].astype(str) + "_" +
            mat[mat.columns[1]].astype(str) + "_" +
            mat[mat.columns[2]].astype(str) + "_" +
            mat[mat.columns[3]].astype(str)
        ).tolist()
    else:
        feature_names = mat[mat.columns[0]].astype(str).tolist()

    X = mat[sample_cols].T.values.astype(np.int8)
    assert X.shape[1] == len(feature_names)
    return X, y, feature_names, sample_cols


def get_param_grid():
    return [
        {"max_depth": 2, "n_estimators": 100, "learning_rate": 0.1},
        {"max_depth": 3, "n_estimators": 200, "learning_rate": 0.1},
        {"max_depth": 4, "n_estimators": 150, "learning_rate": 0.05},
    ]


def train_one_fold(X_tr, y_tr, X_te, y_te, params, strategy):
    """
    Train a single model per strategy.
    'strategy' in {"baseline", "weight", "smote"}.
    """
    k = min(K_BEST, X_tr.shape[1])
    sel = SelectKBest(f_classif, k=k).fit(X_tr, y_tr)
    X_tr_s = sel.transform(X_tr)
    X_te_s = sel.transform(X_te)

    if strategy == "smote":
        sm = SMOTE(k_neighbors=SMOTE_K_NEIGHBORS, random_state=RANDOM_STATE)
        X_tr_s, y_tr_s = sm.fit_resample(X_tr_s, y_tr)
        clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                            random_state=RANDOM_STATE, **params)
    elif strategy == "weight":
        # scale_pos_weight = n_negative / n_positive
        n_pos = int(y_tr.sum())
        n_neg = int((y_tr == 0).sum())
        spw = n_neg / n_pos if n_pos > 0 else 1.0
        clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                            random_state=RANDOM_STATE,
                            scale_pos_weight=spw, **params)
        y_tr_s = y_tr
    elif strategy == "baseline":
        clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                            random_state=RANDOM_STATE, **params)
        y_tr_s = y_tr
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    clf.fit(X_tr_s, y_tr_s)
    proba = clf.predict_proba(X_te_s)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return proba, pred


def inner_cv_select_params(X_tr, y_tr, strategy):
    """Run inner CV to select hyperparameters for the given strategy."""
    inner = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True,
                            random_state=RANDOM_STATE)
    best_auc, best_params = -1, get_param_grid()[0]

    for params in get_param_grid():
        inner_aucs = []
        for in_tr, in_va in inner.split(X_tr, y_tr):
            Xi, Xv = X_tr[in_tr], X_tr[in_va]
            yi, yv = y_tr[in_tr], y_tr[in_va]
            try:
                proba, _ = train_one_fold(Xi, yi, Xv, yv, params, strategy)
                inner_aucs.append(roc_auc_score(yv, proba))
            except Exception:
                inner_aucs.append(np.nan)
        mean_auc = np.nanmean(inner_aucs) if inner_aucs else -1
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_params = params

    return best_params


def evaluate_strategy(X, y, strategy, label):
    """Full nested CV for a given strategy."""
    print(f"\n{'='*60}\n[RUN] {label}\n{'='*60}")
    outer = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True,
                            random_state=RANDOM_STATE)
    fold_metrics, all_yt, all_yp = [], [], []

    for fold_i, (tr, te) in enumerate(outer.split(X, y), 1):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        # Hyperparameter tuning
        best_params = inner_cv_select_params(X_tr, y_tr, strategy)

        # Train final model on full outer-train
        proba, pred = train_one_fold(X_tr, y_tr, X_te, y_te,
                                      best_params, strategy)

        all_yt.extend(y_te); all_yp.extend(proba)
        fold_metrics.append({
            "fold": fold_i,
            "auc": roc_auc_score(y_te, proba),
            "pr_auc": average_precision_score(y_te, proba),
            "accuracy": accuracy_score(y_te, pred),
            "balanced_accuracy": balanced_accuracy_score(y_te, pred),
            "precision": precision_score(y_te, pred, zero_division=0),
            "recall": recall_score(y_te, pred, zero_division=0),
            "f1": f1_score(y_te, pred, zero_division=0),
            "mcc": matthews_corrcoef(y_te, pred),
        })
        print(f"  fold {fold_i}: AUC={fold_metrics[-1]['auc']:.3f} "
              f"F1={fold_metrics[-1]['f1']:.3f} "
              f"MCC={fold_metrics[-1]['mcc']:.3f}")

    df = pd.DataFrame(fold_metrics)
    agg = df.drop(columns="fold").agg(["mean", "std"]).T
    agg.columns = ["mean", "std"]

    # Bootstrap 95% CI for AUC and F1
    yt = np.array(all_yt); yp = np.array(all_yp)
    rng = np.random.default_rng(RANDOM_STATE)

    def bootstrap_ci(y_true, y_pred_prob, metric_fn, n_boot=1000):
        vals = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(y_true), len(y_true))
            if len(np.unique(y_true[idx])) < 2:
                continue
            try:
                vals.append(metric_fn(y_true[idx], y_pred_prob[idx]))
            except Exception:
                continue
        if not vals:
            return np.nan, np.nan
        return np.percentile(vals, [2.5, 97.5])

    auc_lo, auc_hi = bootstrap_ci(yt, yp, roc_auc_score)
    agg.loc["auc", "ci95_lo"] = auc_lo
    agg.loc["auc", "ci95_hi"] = auc_hi

    # F1 CI via thresholding
    def f1_from_proba(y_true, y_prob, thr=0.5):
        return f1_score(y_true, (y_prob >= thr).astype(int), zero_division=0)

    f1_lo, f1_hi = bootstrap_ci(yt, yp, f1_from_proba)
    agg.loc["f1", "ci95_lo"] = f1_lo
    agg.loc["f1", "ci95_hi"] = f1_hi

    print(f"[RESULT] {label}:")
    print(f"  AUC = {agg.loc['auc','mean']:.3f} "
          f"(95% CI {auc_lo:.3f}-{auc_hi:.3f})")
    print(f"  F1  = {agg.loc['f1','mean']:.3f} "
          f"(95% CI {f1_lo:.3f}-{f1_hi:.3f})")
    print(f"  MCC = {agg.loc['mcc','mean']:.3f}")

    return agg, df


def main():
    X, y, feature_names, sample_cols = load_data()
    print(f"[INFO] X shape: {X.shape}")

    results = {}
    for strategy, label in [
        ("baseline", "A_baseline"),
        ("weight", "B_class_weight"),
        ("smote", "C_smote"),
    ]:
        agg, df = evaluate_strategy(X, y, strategy, label)
        agg.to_csv(os.path.join(OUT_DIR, f"metrics_{label}.tsv"), sep="\t")
        df.to_csv(os.path.join(OUT_DIR, f"folds_{label}.tsv"),
                  sep="\t", index=False)
        results[label] = agg

    # Summary comparison
    rows = []
    for label, agg in results.items():
        rows.append({
            "strategy": label,
            "auc_mean": agg.loc["auc", "mean"],
            "auc_ci_lo": agg.loc["auc", "ci95_lo"],
            "auc_ci_hi": agg.loc["auc", "ci95_hi"],
            "f1_mean": agg.loc["f1", "mean"],
            "f1_ci_lo": agg.loc["f1", "ci95_lo"],
            "f1_ci_hi": agg.loc["f1", "ci95_hi"],
            "mcc_mean": agg.loc["mcc", "mean"],
            "precision": agg.loc["precision", "mean"],
            "recall": agg.loc["recall", "mean"],
            "balanced_accuracy": agg.loc["balanced_accuracy", "mean"],
            "pr_auc": agg.loc["pr_auc", "mean"],
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT_DIR, "summary_comparison.tsv"),
                   sep="\t", index=False)

    print("\n" + "="*60)
    print("IMBALANCE HANDLING COMPARISON")
    print("="*60)
    print(summary.to_string(index=False))
    print(f"\n[SAVED] {OUT_DIR}/")


if __name__ == "__main__":
    main()