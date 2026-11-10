"""
XGBoost + SMOTE با DMatrix (کارآمد برای حافظه)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import time
import gc

MATRIX_FILE = Path("data/processed/final_matrix_light.tsv")
PHENO_FILE = Path("data/processed/phenotype_light.tsv")
OUT_DIR = Path("results/xgboost_combined")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Loading data...")
print("=" * 60)

# خواندن header
with open(MATRIX_FILE) as f:
    header = f.readline().strip().split("\t")
feature_names = header[1:]
print(f"Features: {len(feature_names)}")

# خواندن ماتریس (بدون dtype برای ایندکس)
combined = pd.read_csv(
    MATRIX_FILE,
    sep="\t",
)
print(f"Raw shape: {combined.shape}")

# ستون اول = نام نمونه
sample_col = combined.columns[0]
combined = combined.set_index(sample_col)
print(f"After set_index: {combined.shape}")

# phenotype
pheno = pd.read_csv(PHENO_FILE, sep="\t", index_col=0)
pheno = pheno.loc[combined.index]
y = pheno["resistant"].values

print(f"Samples: {len(y)}")
print(f"Resistant: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
print(f"Susceptible: {(1-y).sum()} ({(1-y).sum()/len(y)*100:.1f}%)")

# ماتریس
X = combined.values.astype(np.float32)
sample_names = combined.index.tolist()
del combined
gc.collect()

print(f"X shape: {X.shape}")

# XGBoost parameters
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 3,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "seed": 42,
    "nthread": 4,
    "tree_method": "hist",
}
NUM_ROUNDS = 300

def fit_predict_fold(X_tr, y_tr, X_te):
    n_min = int(y_tr.sum())
    if n_min < 2:
        return None
    k = min(5, n_min - 1)
    smote = SMOTE(random_state=42, k_neighbors=k)
    X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
    
    # DMatrix
    dtrain = xgb.DMatrix(X_tr_res, label=y_tr_res)
    dtest = xgb.DMatrix(X_te)
    
    bst = xgb.train(XGB_PARAMS, dtrain, num_boost_round=NUM_ROUNDS)
    proba = bst.predict(dtest)
    return bst, proba

print("\n" + "=" * 60)
print("Running Leave-One-Out CV...")
print("=" * 60)

loo = LeaveOneOut()
oof_proba = np.full(len(y), np.nan)
start_time = time.time()

for i, (tr_idx, te_idx) in enumerate(loo.split(X)):
    if i % 20 == 0:
        elapsed = time.time() - start_time
        print(f"  Fold {i+1}/{len(y)} (elapsed: {elapsed/60:.1f} min)")
    
    res = fit_predict_fold(X[tr_idx], y[tr_idx], X[te_idx])
    if res is None:
        continue
    _, proba = res
    oof_proba[te_idx[0]] = proba[0]

mask = ~np.isnan(oof_proba)
y_true = y[mask]
y_proba = oof_proba[mask]
y_pred = (y_proba >= 0.5).astype(int)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
metrics = {
    "n_evaluated": int(mask.sum()),
    "n_resistant": int(y_true.sum()),
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "precision": float(precision_score(y_true, y_pred, zero_division=0)),
    "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_true, y_proba)),
    "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
}

print("\n" + "=" * 60)
print("XGBoost + SMOTE (SNP + k-mer) Results:")
print("=" * 60)
for k, v in metrics.items():
    print(f"{k:15s}: {v}")

pd.Series(metrics).to_csv(OUT_DIR / "metrics.tsv", sep="\t")
pd.DataFrame({
    "sample": [sample_names[i] for i in range(len(mask)) if mask[i]],
    "y_true": y_true,
    "y_proba": y_proba,
    "y_pred": y_pred,
}).to_csv(OUT_DIR / "predictions.tsv", sep="\t", index=False)

print(f"\n✓ Saved: {OUT_DIR}")
print("Done!")