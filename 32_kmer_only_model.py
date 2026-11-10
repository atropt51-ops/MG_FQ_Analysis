"""
32_kmer_only_model.py

Evaluate XGBoost using k-mer features ONLY (no SNPs).
Compare against SNP-only and SNP+k-mer.

k-mer matrix structure: rows = samples, cols = k-mers (binary 0/1).
First column (unnamed) holds sample IDs (SRR...).
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

SNP_FILE = "data/processed/snp_matrix_binary.tsv"
KMER_FILE = "results/kmer_python/kmer_matrix_small.tsv"
PHENO_FILE = "data/processed/phenotype_clean.tsv"
OUT_DIR = "results/kmer_only_model"
os.makedirs(OUT_DIR, exist_ok=True)

ACCESSION_COL = "SRR/ERS_Accession "
PHENO_COL = "fluoroquinolone_phenotype"
RESISTANT_VALUE = "R"


def load_phenotype():
    print(f"[INFO] Loading phenotype: {PHENO_FILE}")
    pheno = pd.read_csv(PHENO_FILE, sep="\t")
    pheno[ACCESSION_COL] = pheno[ACCESSION_COL].astype(str).str.strip()
    return pheno.set_index(ACCESSION_COL)


def load_snps(pheno):
    print(f"[INFO] Loading SNP matrix: {SNP_FILE}")
    mat = pd.read_csv(SNP_FILE, sep="\t")
    print(f"[INFO] SNP matrix shape: {mat.shape}")

    sample_cols = list(mat.columns[4:])
    pheno_aligned = pheno.reindex(sample_cols)
    y = (pheno_aligned[PHENO_COL] == RESISTANT_VALUE).astype(int).values

    feature_names = (
        mat[mat.columns[0]].astype(str) + "_" +
        mat[mat.columns[1]].astype(str) + "_" +
        mat[mat.columns[2]].astype(str) + "_" +
        mat[mat.columns[3]].astype(str)
    ).tolist()

    X = mat[sample_cols].T.values.astype(np.int8)
    print(f"[INFO] SNP X shape: {X.shape}, R={int(y.sum())}, S={int((y==0).sum())}")
    return X, y, feature_names, sample_cols


def load_kmers(pheno, sample_cols_snp):
    print(f"[INFO] Loading k-mer matrix: {KMER_FILE}")
    km = pd.read_csv(KMER_FILE, sep="\t")
    print(f"[INFO] k-mer matrix shape: {km.shape}")

    first_col = km.columns[0]
    first_vals = km[first_col].astype(str).tolist()
    print(f"[INFO] First col '{first_col}', example: {first_vals[:3]}")

    if not any(v.startswith(("SRR", "ERS")) for v in first_vals[:5]):
        raise ValueError(f"First col does not contain sample IDs: {first_vals[:3]}")

    print("[INFO] Orientation: rows = samples, cols = k-mers")
    km = km.set_index(first_col)
    km.index = km.index.astype(str).str.strip()

    matched = [s for s in sample_cols_snp if s in km.index]
    print(f"[INFO] Matched samples: {len(matched)} / {len(sample_cols_snp)}")
    if not matched:
        raise ValueError("No samples matched")

    X_k = km.loc[matched].values.astype(np.int8)
    feature_names_k = km.columns.astype(str).tolist()
    print(f"[INFO] k-mer X shape: {X_k.shape}")
    return X_k, feature_names_k, matched


def load_data():
    pheno = load_phenotype()
    X_snp, y_snp, feat_snp, samples_snp = load_snps(pheno)
    X_k, feat_k, samples_k = load_kmers(pheno, samples_snp)

    common = [s for s in samples_snp if s in samples_k]
    print(f"[INFO] Common samples: {len(common)}")
    if len(common) < 50:
        raise ValueError(f"Too few common samples: {len(common)}")

    idx_snp = [samples_snp.index(s) for s in common]
    idx_k = [samples_k.index(s) for s in common]

    X_snp_c = X_snp[idx_snp]
    X_k_c = X_k[idx_k]
    y_c = y_snp[idx_snp]

    print(f"[INFO] Aligned SNP: {X_snp_c.shape}")
    print(f"[INFO] Aligned k-mer: {X_k_c.shape}")
    print(f"[INFO] R={int(y_c.sum())}, S={int((y_c==0).sum())}")

    return X_snp_c, X_k_c, y_c, feat_snp, feat_k


def filter_monomorphic(X, min_alt=3):
    n_alt = X.sum(axis=0)
    keep = n_alt >= min_alt
    print(f"[INFO] Features before filter: {X.shape[1]}, kept: {int(keep.sum())}")
    return X[:, keep]


def get_param_grid():
    return [
        {"max_depth": 2, "n_estimators": 100, "learning_rate": 0.1},
        {"max_depth": 3, "n_estimators": 200, "learning_rate": 0.1},
        {"max_depth": 4, "n_estimators": 150, "learning_rate": 0.05},
    ]


def train_and_eval_fold(X_tr, y_tr, X_te, y_te, params):
    k = min(K_BEST, X_tr.shape[1])
    sel = SelectKBest(f_classif, k=k).fit(X_tr, y_tr)
    X_tr_s = sel.transform(X_tr)
    X_te_s = sel.transform(X_te)

    sm = SMOTE(k_neighbors=SMOTE_K_NEIGHBORS, random_state=RANDOM_STATE)
    X_tr_s, y_tr_s = sm.fit_resample(X_tr_s, y_tr)

    clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                        random_state=RANDOM_STATE, **params)
    clf.fit(X_tr_s, y_tr_s)
    proba = clf.predict_proba(X_te_s)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return proba, pred


def inner_cv_select_params(X_tr, y_tr):
    inner = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True,
                            random_state=RANDOM_STATE)
    best_auc, best_params = -1, get_param_grid()[0]
    for params in get_param_grid():
        aucs = []
        for in_tr, in_va in inner.split(X_tr, y_tr):
            try:
                proba, _ = train_and_eval_fold(
                    X_tr[in_tr], y_tr[in_tr],
                    X_tr[in_va], y_tr[in_va], params)
                aucs.append(roc_auc_score(y_tr[in_va], proba))
            except Exception:
                aucs.append(np.nan)
        m = np.nanmean(aucs) if aucs else -1
        if m > best_auc:
            best_auc, best_params = m, params
    return best_params


def evaluate(X, y, label):
    print(f"\n{'='*60}\n[RUN] {label}\n{'='*60}")
    print(f"[INFO] Features: {X.shape[1]}")
    X = filter_monomorphic(X)

    outer = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True,
                            random_state=RANDOM_STATE)
    fold_metrics, all_yt, all_yp = [], [], []

    for fold_i, (tr, te) in enumerate(outer.split(X, y), 1):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        best_params = inner_cv_select_params(X_tr, y_tr)
        proba, pred = train_and_eval_fold(X_tr, y_tr, X_te, y_te, best_params)

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

    df = pd.DataFrame(fold_metrics)
    agg = df.drop(columns="fold").agg(["mean", "std"]).T
    agg.columns = ["mean", "std"]

    yt = np.array(all_yt); yp = np.array(all_yp)
    rng = np.random.default_rng(RANDOM_STATE)
    boot = []
    for _ in range(1000):
        idx = rng.integers(0, len(yt), len(yt))
        if len(np.unique(yt[idx])) < 2:
            continue
        boot.append(roc_auc_score(yt[idx], yp[idx]))
    if boot:
        lo, hi = np.percentile(boot, [2.5, 97.5])
        agg.loc["auc", "ci95_lo"] = lo
        agg.loc["auc", "ci95_hi"] = hi
        print(f"[RESULT] {label}: AUC = {agg.loc['auc','mean']:.3f} "
              f"(95% CI {lo:.3f}-{hi:.3f})")
    return agg, df


def main():
    X_snp, X_kmer, y, feat_snp, feat_k = load_data()

    results = {}

    agg_snp, df_snp = evaluate(X_snp, y, "SNP_only")
    agg_snp.to_csv(os.path.join(OUT_DIR, "metrics_snp_only.tsv"), sep="\t")
    df_snp.to_csv(os.path.join(OUT_DIR, "folds_snp_only.tsv"),
                  sep="\t", index=False)
    results["SNP_only"] = (agg_snp, X_snp.shape[1])

    agg_k, df_k = evaluate(X_kmer, y, "kmer_only")
    agg_k.to_csv(os.path.join(OUT_DIR, "metrics_kmer_only.tsv"), sep="\t")
    df_k.to_csv(os.path.join(OUT_DIR, "folds_kmer_only.tsv"),
                sep="\t", index=False)
    results["kmer_only"] = (agg_k, X_kmer.shape[1])

    X_comb = np.hstack([X_snp, X_kmer])
    print(f"\n[INFO] Combined: {X_comb.shape}")
    agg_c, df_c = evaluate(X_comb, y, "SNP_plus_kmer")
    agg_c.to_csv(os.path.join(OUT_DIR, "metrics_snp_plus_kmer.tsv"), sep="\t")
    df_c.to_csv(os.path.join(OUT_DIR, "folds_snp_plus_kmer.tsv"),
                sep="\t", index=False)
    results["SNP_plus_kmer"] = (agg_c, X_comb.shape[1])

    rows = []
    for name, (agg, nf) in results.items():
        rows.append({
            "model": name,
            "n_features": nf,
            "auc_mean": agg.loc["auc", "mean"],
            "auc_ci_lo": agg.loc["auc", "ci95_lo"],
            "auc_ci_hi": agg.loc["auc", "ci95_hi"],
            "f1_mean": agg.loc["f1", "mean"],
            "mcc_mean": agg.loc["mcc", "mean"],
            "precision": agg.loc["precision", "mean"],
            "recall": agg.loc["recall", "mean"],
            "pr_auc": agg.loc["pr_auc", "mean"],
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT_DIR, "summary_comparison.tsv"),
                   sep="\t", index=False)

    print("\n" + "="*60)
    print("k-mer vs SNP COMPARISON")
    print("="*60)
    print(summary.to_string(index=False))
    print(f"\n[SAVED] {OUT_DIR}/")


if __name__ == "__main__":
    main()