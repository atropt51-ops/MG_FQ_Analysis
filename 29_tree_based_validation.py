"""
29_tree_based_validation.py

Lineage-aware validation of the XGBoost classifier.

Three validation strategies:
  1) GroupKFold by BAPS_AUS_group (6 groups) - strictest
  2) GroupKFold by Major_Clade (2 groups) - coarse
  3) GroupKFold by BAPS_AUS_group after removing the mgpA haplotype
     (positions 222607-228518), to test whether performance is
     driven by the highly polymorphic adhesin P1 locus.

Motivation: standard LOOCV/StratifiedKFold allows genetically similar
isolates from the same lineage to appear in both training and test folds,
inflating performance. GroupKFold forces test samples to belong to
lineages unseen during training, testing true generalization.

Inputs:
  data/processed/snp_matrix_binary.tsv
  data/processed/phenotype_clean.tsv

Outputs:
  results/tree_validation/
    metrics_gkf_baps.tsv
    metrics_gkf_clade.tsv
    metrics_gkf_baps_no_mgpA.tsv
    summary_comparison.tsv
    removed_mgpA_features.txt
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, average_precision_score,
    balanced_accuracy_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
RANDOM_STATE = 42
N_FOLDS = 5
N_INNER_FOLDS = 3
K_BEST = 100
SMOTE_K_NEIGHBORS = 5

MATRIX_FILE = "data/processed/snp_matrix_binary.tsv"
PHENO_FILE = "data/processed/phenotype_clean.tsv"
OUT_DIR = "results/tree_validation"
os.makedirs(OUT_DIR, exist_ok=True)

ACCESSION_COL = "SRR/ERS_Accession "
PHENO_COL = "fluoroquinolone_phenotype"
RESISTANT_VALUE = "R"

# mgpA haplotype region (from GFF, includes all 20 linked SNPs)
MGPA_START = 222_607
MGPA_END = 228_518


def load_data():
    print(f"[INFO] Loading matrix: {MATRIX_FILE}")
    mat = pd.read_csv(MATRIX_FILE, sep="\t")
    print(f"[INFO] Matrix shape: {mat.shape}")

    print(f"[INFO] Loading phenotype: {PHENO_FILE}")
    pheno = pd.read_csv(PHENO_FILE, sep="\t")
    pheno[ACCESSION_COL] = pheno[ACCESSION_COL].astype(str).str.strip()

    sample_cols = list(mat.columns[4:])
    pheno = pheno.set_index(ACCESSION_COL).reindex(sample_cols)

    y = (pheno[PHENO_COL] == RESISTANT_VALUE).astype(int).values
    print(f"[INFO] Class balance: R={int(y.sum())}, S={int((y==0).sum())}")

    # Feature IDs
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

    # Groups
    groups_baps = pheno["BAPS_AUS_group"].fillna(-1).astype(int).values
    groups_clade = pheno["Major_Clade"].fillna("NA").astype(str).values

    print(f"[INFO] BAPS_AUS_group distribution: "
          f"{dict(pd.Series(groups_baps).value_counts())}")
    print(f"[INFO] Major_Clade distribution: "
          f"{dict(pd.Series(groups_clade).value_counts())}")

    return X, y, feature_names, groups_baps, groups_clade


def identify_mgpA_features(feature_names):
    """Return indices of features in the mgpA region."""
    indices = []
    for i, name in enumerate(feature_names):
        m = re.search(r"_(\d+)_", name)
        if m:
            pos = int(m.group(1))
            if MGPA_START <= pos <= MGPA_END:
                indices.append(i)
    return set(indices)


def group_cv_evaluate(X, y, groups, label, n_splits=5):
    """
    GroupKFold evaluation. Test samples always belong to a group
    not present in the training fold.
    """
    print(f"\n{'='*60}\n[RUN] {label}\n{'='*60}")
    print(f"[INFO] Groups: {len(np.unique(groups))}, samples: {len(y)}")

    # Use GroupKFold; fall back to fewer splits if some group is tiny
    unique_groups, counts = np.unique(groups, return_counts=True)
    n_groups = len(unique_groups)
    actual_splits = min(n_splits, n_groups)
    if actual_splits < n_splits:
        print(f"[WARN] Only {n_groups} groups, using {actual_splits} folds")

    gkf = GroupKFold(n_splits=actual_splits)
    fold_metrics, all_yt, all_yp = [], [], []

    param_grid = [
        {"max_depth": 2, "n_estimators": 100, "learning_rate": 0.1},
        {"max_depth": 3, "n_estimators": 200, "learning_rate": 0.1},
        {"max_depth": 4, "n_estimators": 150, "learning_rate": 0.05},
    ]

    for fold_i, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        if len(np.unique(y[te])) < 2:
            print(f"  fold {fold_i}: skipped (single class in test)")
            continue

        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        # Inner CV: StratifiedKFold within training (group constraint loose here
        # because inner folds only tune hyperparameters, not generalization)
        inner = StratifiedKFold(n_splits=min(N_INNER_FOLDS, np.bincount(y_tr).min()),
                                shuffle=True, random_state=RANDOM_STATE)
        best_auc, best_params = -1, param_grid[0]
        for params in param_grid:
            inner_aucs = []
            for in_tr, in_va in inner.split(X_tr, y_tr):
                Xi, Xv = X_tr[in_tr], X_tr[in_va]
                yi, yv = y_tr[in_tr], y_tr[in_va]
                k = min(K_BEST, Xi.shape[1])
                sel = SelectKBest(f_classif, k=k).fit(Xi, yi)
                Xi = sel.transform(Xi); Xv = sel.transform(Xv)
                Xi, yi = SMOTE(k_neighbors=SMOTE_K_NEIGHBORS,
                               random_state=RANDOM_STATE).fit_resample(Xi, yi)
                clf = XGBClassifier(use_label_encoder=False,
                                    eval_metric="logloss",
                                    random_state=RANDOM_STATE, **params)
                clf.fit(Xi, yi)
                inner_aucs.append(roc_auc_score(yv, clf.predict_proba(Xv)[:, 1]))
            if np.mean(inner_aucs) > best_auc:
                best_auc, best_params = np.mean(inner_aucs), params

        # Train final model on the full training group
        k = min(K_BEST, X_tr.shape[1])
        sel = SelectKBest(f_classif, k=k).fit(X_tr, y_tr)
        X_tr_s = sel.transform(X_tr); X_te_s = sel.transform(X_te)
        X_tr_s, y_tr_s = SMOTE(k_neighbors=SMOTE_K_NEIGHBORS,
                               random_state=RANDOM_STATE).fit_resample(X_tr_s, y_tr)
        clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                            random_state=RANDOM_STATE, **best_params)
        clf.fit(X_tr_s, y_tr_s)
        proba = clf.predict_proba(X_te_s)[:, 1]
        pred = (proba >= 0.5).astype(int)

        all_yt.extend(y_te); all_yp.extend(proba)
        fold_metrics.append({
            "fold": fold_i,
            "n_test": len(y_te),
            "groups_in_test": ",".join(map(str, np.unique(groups[te]))),
            "auc": roc_auc_score(y_te, proba),
            "pr_auc": average_precision_score(y_te, proba),
            "accuracy": accuracy_score(y_te, pred),
            "balanced_accuracy": balanced_accuracy_score(y_te, pred),
            "precision": precision_score(y_te, pred, zero_division=0),
            "recall": recall_score(y_te, pred, zero_division=0),
            "f1": f1_score(y_te, pred, zero_division=0),
            "mcc": matthews_corrcoef(y_te, pred),
        })
        print(f"  fold {fold_i} (groups {fold_metrics[-1]['groups_in_test']}): "
              f"AUC={fold_metrics[-1]['auc']:.3f} F1={fold_metrics[-1]['f1']:.3f}")

    if not fold_metrics:
        print("[ERROR] No valid folds.")
        return None

    df = pd.DataFrame(fold_metrics)
    agg = df.drop(columns=["fold", "n_test", "groups_in_test"]).agg(
        ["mean", "std"]).T
    agg.columns = ["mean", "std"]

    # Bootstrap CI on pooled predictions
    if len(all_yt) > 0 and len(np.unique(all_yt)) >= 2:
        yt = np.array(all_yt); yp = np.array(all_yp)
        rng = np.random.default_rng(RANDOM_STATE)
        boot = []
        for _ in range(1000):
            idx = rng.integers(0, len(yt), len(yt))
            if len(np.unique(yt[idx])) < 2:
                continue
            boot.append(roc_auc_score(yt[idx], yp[idx]))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        agg.loc["auc", "ci95_lo"] = lo
        agg.loc["auc", "ci95_hi"] = hi
        print(f"[RESULT] {label}: AUC = {agg.loc['auc','mean']:.3f} "
              f"(95% CI {lo:.3f}-{hi:.3f})")
    else:
        print(f"[RESULT] {label}: AUC = {agg.loc['auc','mean']:.3f}")

    return agg, df


def main():
    X, y, feature_names, groups_baps, groups_clade = load_data()
    print(f"[INFO] X shape: {X.shape}")

    # Identify mgpA region
    mgpA_idx = identify_mgpA_features(feature_names)
    print(f"[INFO] mgpA-region features (222607-228518): {len(mgpA_idx)}")

    # Save list of removed mgpA features for manuscript
    with open(os.path.join(OUT_DIR, "removed_mgpA_features.txt"), "w") as f:
        f.write(f"# mgpA haplotype region: {MGPA_START}-{MGPA_END}\n")
        f.write(f"# Total features removed: {len(mgpA_idx)}\n")
        for i in sorted(mgpA_idx):
            f.write(feature_names[i] + "\n")

    results = {}

    # 1) GroupKFold by BAPS_AUS_group
    r = group_cv_evaluate(X, y, groups_baps, "GKFBAPS_all_features")
    if r:
        agg, df = r
        agg.to_csv(os.path.join(OUT_DIR, "metrics_gkf_baps.tsv"), sep="\t")
        df.to_csv(os.path.join(OUT_DIR, "folds_gkf_baps.tsv"),
                  sep="\t", index=False)
        results["gkf_baps"] = agg

    # 2) GroupKFold by Major_Clade
    r = group_cv_evaluate(X, y, groups_clade, "GKFClade_all_features")
    if r:
        agg, df = r
        agg.to_csv(os.path.join(OUT_DIR, "metrics_gkf_clade.tsv"), sep="\t")
        df.to_csv(os.path.join(OUT_DIR, "folds_gkf_clade.tsv"),
                  sep="\t", index=False)
        results["gkf_clade"] = agg

    # 3) GroupKFold by BAPS_AUS_group, no mgpA region
    keep = [i for i in range(len(feature_names)) if i not in mgpA_idx]
    X_nomgpA = X[:, keep]
    print(f"\n[INFO] Features after removing mgpA: {X_nomgpA.shape[1]}")
    r = group_cv_evaluate(X_nomgpA, y, groups_baps,
                          "GKFBAPS_no_mgpA")
    if r:
        agg, df = r
        agg.to_csv(os.path.join(OUT_DIR, "metrics_gkf_baps_no_mgpA.tsv"),
                   sep="\t")
        df.to_csv(os.path.join(OUT_DIR, "folds_gkf_baps_no_mgpA.tsv"),
                  sep="\t", index=False)
        results["gkf_baps_no_mgpA"] = agg

    # Summary
    rows = []
    for name, agg in results.items():
        rows.append({
            "strategy": name,
            "auc_mean": agg.loc["auc", "mean"],
            "auc_ci_lo": agg.loc["auc", "ci95_lo"],
            "auc_ci_hi": agg.loc["auc", "ci95_hi"],
            "f1_mean": agg.loc["f1", "mean"],
            "mcc_mean": agg.loc["mcc", "mean"],
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT_DIR, "summary_comparison.tsv"),
                   sep="\t", index=False)

    print("\n" + "="*60)
    print("LINEAGE-AWARE VALIDATION SUMMARY")
    print("="*60)
    print(summary.to_string(index=False))
    print(f"\n[SAVED] {OUT_DIR}/")


if __name__ == "__main__":
    main()