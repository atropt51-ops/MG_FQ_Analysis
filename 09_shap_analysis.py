"""
SHAP analysis برای تفسیر مدل XGBoost
"""
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt

MATRIX_FILE = Path("data/processed/snp_matrix_binary.tsv")
PHENO_FILE = Path("data/processed/phenotype_clean.tsv")
OUT_DIR = Path("results/shap_azzato")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading data...")
mat = pd.read_csv(MATRIX_FILE, sep="\t")
pheno = pd.read_csv(PHENO_FILE, sep="\t")

accession_col = "SRR/ERS_Accession "
pheno[accession_col] = pheno[accession_col].astype(str).str.strip()
sample_cols = mat.columns[4:]

# Feature names از سطرها (SNPها)
feature_names = (
    mat["CHROM"].astype(str) + "_" + 
    mat["POS"].astype(str) + "_" + 
    mat["REF"].astype(str) + "_" + 
    mat["ALT"].astype(str)
).tolist()

print(f"feature_names count: {len(feature_names)}")
print(f"First 5: {feature_names[:5]}")

# هم‌ترازی phenotype
pheno = pheno.set_index(accession_col).loc[sample_cols]
y = (pheno["fluoroquinolone_phenotype"] == "R").astype(int).values

# ماتریس X: samples × SNPs (transpose)
X = mat[sample_cols].T.values.astype(np.int8)

print(f"X shape: {X.shape}, y shape: {y.shape}")

# Train XGBoost
print("\nTraining XGBoost on full data...")
XGB_PARAMS = dict(
    n_estimators=500, max_depth=3, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", random_state=42, n_jobs=-1, tree_method="hist",
)

smote = SMOTE(random_state=42, k_neighbors=5)
X_res, y_res = smote.fit_resample(X, y)
print(f"After SMOTE: {X_res.shape}")

clf = XGBClassifier(**XGB_PARAMS)
clf.fit(X_res, y_res)

# SHAP
print("\nComputing SHAP values...")
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X)

print(f"shap_values shape: {shap_values.shape}")

# ذخیره
np.save(OUT_DIR / "shap_values.npy", shap_values)
np.save(OUT_DIR / "X.npy", X)
np.save(OUT_DIR / "y.npy", y)

with open(OUT_DIR / "feature_names.txt", "w") as f:
    for name in feature_names:
        f.write(f"{name}\n")

# Summary plot
print("\nGenerating SHAP summary plot...")
plt.figure(figsize=(10, 12))
shap.summary_plot(shap_values, X, feature_names=feature_names, max_display=20, show=False)
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

# Bar plot
print("Generating SHAP bar plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", max_display=20, show=False)
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()

# Top features
print("\nComputing top features...")
mean_shap = np.abs(shap_values).mean(axis=0)

top_features = pd.DataFrame({
    "feature": feature_names,
    "mean_shap": mean_shap,
}).sort_values("mean_shap", ascending=False)

top_features.to_csv(OUT_DIR / "shap_importance.tsv", sep="\t", index=False)

print("\n=== Top 20 SHAP Features ===")
print(top_features.head(20).to_string(index=False))

print(f"\n✓ Saved to {OUT_DIR}")