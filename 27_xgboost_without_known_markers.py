"""
27_xgboost_without_known_markers.py

Sensitivity analysis: does the model retain predictive power after
removing known fluoroquinolone resistance markers?

Coordinates extracted from data/reference/G37.gff3:
  parC: 242224-244569 (+ strand)
  gyrA:  4812- 7322 (+ strand)

Confirmed QRDR SNPs in snp_matrix_binary.tsv:
  parC codon 83: NC_000908.2_242470_A_T,C  +  NC_000908.2_242471_G_T
  parC codon 87: NC_000908.2_242482_G_A
  gyrA codon 95: NC_000908.2_5096_G_A,T
  (gyrA codons 83 and 87 are absent from this dataset)

Three scenarios:
  A) Full model
  B) Remove parC S83I (codon 83)
  C) Remove full QRDR: parC 83,87 + gyrA 95
"""

import os
import re
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

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
RANDOM_STATE = 42
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3
K_BEST = 100
SMOTE_K_NEIGHBORS = 5

MATRIX_FILE = "data/processed/snp_matrix_binary.tsv"
PHENO_FILE = "data/processed/phenotype_clean.tsv"
OUT_DIR = "results/xgboost_sensitivity"
os.makedirs(OUT_DIR, exist_ok=True)

ACCESSION_COL = "SRR/ERS_Accession "   # trailing space is intentional
PHENO_COL = "fluoroquinolone_phenotype"
RESISTANT_VALUE = "R"

# --- Gene coordinates from G37.gff3 (verified) ---
GENES = {
    "parC": {"start": 242224, "end": 244569, "strand": "+"},
    "gyrA": {"start": 4812,   "end": 7322,   "strand": "+"},
}

# Codon numbers of interest
PARC_CODONS = {"S83I": 83, "D87N": 87}
GYRA_CODONS = {"83": 83, "87": 87, "M95I": 95}


def codon_to_nt_range(gene_start, gene_end, strand, codon_num):
    """Return inclusive genomic range (lo, hi) for a given codon."""
    if strand == "+":
        offset = (codon_num - 1) * 3
        lo = gene_start + offset
        hi = lo + 2
    else:  # "-" strand
        offset = (codon_num - 1) * 3
        hi = gene_end - offset
        lo = hi - 2
    return lo, hi


def load_data():
    print(f"[INFO] Loading SNP matrix: {MATRIX_FILE}")
    mat = pd.read_csv(MATRIX_FILE, sep="\t")
    print(f"[INFO] Matrix shape: {mat.shape}")
    print(f"[INFO] First 6 columns: {list(mat.columns[:6])}")

    print(f"[INFO] Loading phenotype: {PHENO_FILE}")
    pheno = pd.read_csv(PHENO_FILE, sep="\t")
    print(f"[INFO] Phenotype shape: {pheno.shape}")

    if ACCESSION_COL not in pheno.columns:
        raise ValueError(
            f"Accession column '{ACCESSION_COL}' not found. "
            f"Available: {list(pheno.columns)}"
        )
    if PHENO_COL not in pheno.columns:
        raise ValueError(
            f"Phenotype column '{PHENO_COL}' not found. "
            f"Available: {list(pheno.columns)}"
        )

    pheno[ACCESSION_COL] = pheno[ACCESSION_COL].astype(str).str.strip()
    sample_cols = list(mat.columns[4:])
    print(f"[INFO] Sample columns in matrix: {len(sample_cols)}")

    pheno = pheno.set_index(ACCESSION_COL)
    missing = [s for s in sample_cols if s not in pheno.index]
    if missing:
        print(f"[WARN] {len(missing)} matrix samples missing from phenotype")
    pheno = pheno.reindex(sample_cols)

    y = (pheno[PHENO_COL] == RESISTANT_VALUE).astype(int).values
    print(f"[INFO] Class balance: R={int(y.sum())}, S={int((y==0).sum())}")

    # Build feature IDs
    first_col = mat.columns[0]
    if first_col.upper() in ("CHROM", "CHR", "#CHROM"):
        feature_names = (
            mat[mat.columns[0]].astype(str) + "_" +
            mat[mat.columns[1]].astype(str) + "_" +
            mat[mat.columns[2]].astype(str) + "_" +
            mat[mat.columns[3]].astype(str)
        ).tolist()
        print(f"[INFO] Built feature IDs from CHROM/POS/REF/ALT")
    else:
        feature_names = mat[first_col].astype(str).tolist()
        print(f"[INFO] Using column '{first_col}' as feature IDs")
    print(f"[INFO] Example IDs: {feature_names[:3]}")

    X = mat[sample_cols].T.values.astype(np.int8)
    assert X.shape[1] == len(feature_names)
    return X, y, feature_names, sample_cols


def identify_known_markers(feature_names):
    """Identify QRDR SNP indices using verified GFF coordinates."""
    def parse_pos(name):
        m = re.search(r"_(\d+)_", name)
        return int(m.group(1)) if m else None

    # Precompute nt ranges
    parC_83 = codon_to_nt_range(GENES["parC"]["start"], GENES["parC"]["end"],
                                 GENES["parC"]["strand"], 83)
    parC_87 = codon_to_nt_range(GENES["parC"]["start"], GENES["parC"]["end"],
                                 GENES["parC"]["strand"], 87)
    gyrA_83_r = codon_to_nt_range(GENES["gyrA"]["start"], GENES["gyrA"]["end"],
                                   GENES["gyrA"]["strand"], 83)
    gyrA_87_r = codon_to_nt_range(GENES["gyrA"]["start"], GENES["gyrA"]["end"],
                                   GENES["gyrA"]["strand"], 87)
    gyrA_95_r = codon_to_nt_range(GENES["gyrA"]["start"], GENES["gyrA"]["end"],
                                   GENES["gyrA"]["strand"], 95)

    print(f"[INFO] parC S83I codon = nt {parC_83[0]}-{parC_83[1]}")
    print(f"[INFO] parC D87N codon = nt {parC_87[0]}-{parC_87[1]}")
    print(f"[INFO] gyrA codon 83  = nt {gyrA_83_r[0]}-{gyrA_83_r[1]}")
    print(f"[INFO] gyrA codon 87  = nt {gyrA_87_r[0]}-{gyrA_87_r[1]}")
    print(f"[INFO] gyrA M95I codon= nt {gyrA_95_r[0]}-{gyrA_95_r[1]}")

    parC_S83I, parC_D87N = set(), set()
    gyrA_83, gyrA_87, gyrA_M95I = set(), set(), set()

    for i, name in enumerate(feature_names):
        pos = parse_pos(name)
        if pos is None:
            continue
        if parC_83[0] <= pos <= parC_83[1]:
            parC_S83I.add(i)
        elif parC_87[0] <= pos <= parC_87[1]:
            parC_D87N.add(i)
        elif gyrA_83_r[0] <= pos <= gyrA_83_r[1]:
            gyrA_83.add(i)
        elif gyrA_87_r[0] <= pos <= gyrA_87_r[1]:
            gyrA_87.add(i)
        elif gyrA_95_r[0] <= pos <= gyrA_95_r[1]:
            gyrA_M95I.add(i)

    return {
        "parC_S83I": parC_S83I,
        "parC_D87N": parC_D87N,
        "gyrA_83": gyrA_83,
        "gyrA_87": gyrA_87,
        "gyrA_M95I": gyrA_M95I,
        "QRDR_all": parC_S83I | parC_D87N | gyrA_83 | gyrA_87 | gyrA_M95I,
    }


def nested_cv_evaluate(X, y, label):
    print(f"\n{'='*60}\n[RUN] {label}\n{'='*60}")
    outer = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True,
                            random_state=RANDOM_STATE)
    fold_metrics, all_yt, all_yp = [], [], []
    param_grid = [
        {"max_depth": 2, "n_estimators": 100, "learning_rate": 0.1},
        {"max_depth": 3, "n_estimators": 200, "learning_rate": 0.1},
        {"max_depth": 4, "n_estimators": 150, "learning_rate": 0.05},
    ]

    for fold_i, (tr, te) in enumerate(outer.split(X, y), 1):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        # Inner CV for hyperparameter selection
        inner = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True,
                                random_state=RANDOM_STATE)
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

        # Train on full outer-train
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
              f"F1={fold_metrics[-1]['f1']:.3f}")

    # Aggregate
    df = pd.DataFrame(fold_metrics)
    agg = df.drop(columns="fold").agg(["mean", "std"]).T
    agg.columns = ["mean", "std"]

    # Bootstrap 95% CI for AUC
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
    return agg, df


def main():
    X, y, feature_names, sample_cols = load_data()
    print(f"[INFO] X shape: {X.shape}, features: {len(feature_names)}")

    markers = identify_known_markers(feature_names)
    print("\n[INFO] Known marker SNP counts:")
    for k, v in markers.items():
        print(f"  {k}: {len(v)} SNPs")
        for i in sorted(v):
            print(f"    - {feature_names[i]}")

    if len(markers["QRDR_all"]) == 0:
        print("\n[ERROR] No QRDR SNPs found — check coordinates.")
        return

    # Save removed-feature report
    rows = []
    for scen, idx_set in [("B_no_parC_S83I", markers["parC_S83I"]),
                          ("C_no_QRDR", markers["QRDR_all"])]:
        for i in sorted(idx_set):
            rows.append({"scenario": scen, "feature_name": feature_names[i]})
    pd.DataFrame(rows).to_csv(
        os.path.join(OUT_DIR, "removed_features_report.tsv"),
        sep="\t", index=False)

    # Scenario A: full
    agg_A, fold_A = nested_cv_evaluate(X, y, "A_full_model")
    agg_A.to_csv(os.path.join(OUT_DIR, "metrics_A_full.tsv"), sep="\t")
    fold_A.to_csv(os.path.join(OUT_DIR, "folds_A_full.tsv"),
                  sep="\t", index=False)

    # Scenario B: remove parC S83I
    keep_B = [i for i in range(len(feature_names))
              if i not in markers["parC_S83I"]]
    agg_B, fold_B = nested_cv_evaluate(X[:, keep_B], y, "B_no_parC_S83I")
    agg_B.to_csv(os.path.join(OUT_DIR, "metrics_B_no_parC_S83I.tsv"), sep="\t")
    fold_B.to_csv(os.path.join(OUT_DIR, "folds_B_no_parC_S83I.tsv"),
                  sep="\t", index=False)

    # Scenario C: remove all QRDR
    keep_C = [i for i in range(len(feature_names))
              if i not in markers["QRDR_all"]]
    agg_C, fold_C = nested_cv_evaluate(X[:, keep_C], y, "C_no_QRDR")
    agg_C.to_csv(os.path.join(OUT_DIR, "metrics_C_no_QRDR.tsv"), sep="\t")
    fold_C.to_csv(os.path.join(OUT_DIR, "folds_C_no_QRDR.tsv"),
                  sep="\t", index=False)

    # Summary comparison
    summary = pd.DataFrame({
        "scenario": ["A_full", "B_no_parC_S83I", "C_no_QRDR"],
        "n_features": [X.shape[1], len(keep_B), len(keep_C)],
        "auc_mean": [agg_A.loc["auc","mean"], agg_B.loc["auc","mean"],
                     agg_C.loc["auc","mean"]],
        "auc_ci_lo": [agg_A.loc["auc","ci95_lo"], agg_B.loc["auc","ci95_lo"],
                      agg_C.loc["auc","ci95_lo"]],
        "auc_ci_hi": [agg_A.loc["auc","ci95_hi"], agg_B.loc["auc","ci95_hi"],
                      agg_C.loc["auc","ci95_hi"]],
        "f1_mean": [agg_A.loc["f1","mean"], agg_B.loc["f1","mean"],
                    agg_C.loc["f1","mean"]],
        "mcc_mean": [agg_A.loc["mcc","mean"], agg_B.loc["mcc","mean"],
                     agg_C.loc["mcc","mean"]],
        "delta_auc_vs_full": [
            0.0,
            agg_B.loc["auc","mean"] - agg_A.loc["auc","mean"],
            agg_C.loc["auc","mean"] - agg_A.loc["auc","mean"],
        ],
    })
    summary.to_csv(os.path.join(OUT_DIR, "summary_comparison.tsv"),
                   sep="\t", index=False)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(summary.to_string(index=False))
    print(f"\n[SAVED] {OUT_DIR}/")


if __name__ == "__main__":
    main()