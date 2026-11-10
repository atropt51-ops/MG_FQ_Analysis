"""
محاسبه λGC، PCA، و QQ plot
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MATRIX_FILE = Path("data/processed/snp_matrix_binary.tsv")
PHENO_FILE = Path("data/processed/phenotype_clean.tsv")
GWAS_FILE = Path("results/gwas_azzato/gwas_results.tsv")
OUT_DIR = Path("results/gwas_diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Loading data...")
print("=" * 60)

# ماتریس SNP
mat = pd.read_csv(MATRIX_FILE, sep="\t")
pheno = pd.read_csv(PHENO_FILE, sep="\t")

accession_col = "SRR/ERS_Accession "
pheno[accession_col] = pheno[accession_col].astype(str).str.strip()
sample_cols = mat.columns[4:]
pheno = pheno.set_index(accession_col).loc[sample_cols]

y = (pheno["fluoroquinolone_phenotype"] == "R").astype(int).values
X = mat[sample_cols].T.values.astype(np.float32)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Resistant: {y.sum()}, Susceptible: {(1-y).sum()}")

# ============================================================
# ۱. محاسبه λGC
# ============================================================
print("\n" + "=" * 60)
print("Step 1: Computing λGC")
print("=" * 60)

# خواندن GWAS
gwas = pd.read_csv(GWAS_FILE, sep="\t")
valid_p = gwas["p_value"].dropna()
valid_p = valid_p[valid_p > 0]

# chi2
chi2_vals = chi2.isf(valid_p.values, df=1)

# λGC = median(chi2) / 0.4549 (chi2.ppf(0.5, df=1))
lambda_gc = np.median(chi2_vals) / chi2.ppf(0.5, df=1)

print(f"λGC = {lambda_gc:.4f}")
print(f"Median chi2 = {np.median(chi2_vals):.4f}")
print(f"Number of valid SNPs: {len(valid_p)}")

# تفسیر
if lambda_gc < 1.0:
    interp = "No inflation (λGC < 1.0)"
elif lambda_gc < 1.1:
    interp = "Mild inflation (1.0 ≤ λGC < 1.1)"
elif lambda_gc < 1.5:
    interp = "Moderate inflation (1.1 ≤ λGC < 1.5)"
else:
    interp = "Severe inflation (λGC ≥ 1.5)"
print(f"Interpretation: {interp}")

# ============================================================
# ۲. PCA
# ============================================================
print("\n" + "=" * 60)
print("Step 2: PCA on SNP matrix")
print("=" * 60)

# استانداردسازی
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=10, random_state=42)
pcs = pca.fit_transform(X_scaled)

print(f"Explained variance (PC1-PC10):")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var*100:.2f}%")

# ذخیره
pc_df = pd.DataFrame(
    pcs,
    index=sample_cols,
    columns=[f"PC{i+1}" for i in range(10)]
)
pc_df["resistant"] = y
pc_df.to_csv(OUT_DIR / "pca_scores.tsv", sep="\t")
print(f"\n✓ Saved: {OUT_DIR / 'pca_scores.tsv'}")

# ============================================================
# ۳. QQ Plot
# ============================================================
print("\n" + "=" * 60)
print("Step 3: Generating QQ plot")
print("=" * 60)

# مرتب‌سازی p-value
sorted_p = np.sort(valid_p.values)
n = len(sorted_p)
expected_p = np.arange(1, n + 1) / n

# QQ plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(-np.log10(expected_p), -np.log10(sorted_p), s=5, alpha=0.5, c="blue")
ax.plot([0, -np.log10(expected_p.min())], [0, -np.log10(expected_p.min())],
        color="red", linestyle="--", label="Expected (null)")
ax.set_xlabel("-log10(Expected P)", fontsize=12)
ax.set_ylabel("-log10(Observed P)", fontsize=12)
ax.set_title(f"QQ Plot (λGC = {lambda_gc:.3f})", fontsize=14)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "qq_plot.png", dpi=150)
plt.close()
print(f"✓ Saved: {OUT_DIR / 'qq_plot.png'}")

# ============================================================
# ۴. ذخیره نتایج
# ============================================================
print("\n" + "=" * 60)
print("Step 4: Saving diagnostics")
print("=" * 60)

diagnostics = pd.DataFrame({
    "metric": ["lambda_gc", "median_chi2", "n_valid_snps", "interpretation",
               "PC1_var", "PC2_var", "PC3_var", "PC4_var", "PC5_var"],
    "value": [lambda_gc, np.median(chi2_vals), len(valid_p), interp,
              pca.explained_variance_ratio_[0],
              pca.explained_variance_ratio_[1],
              pca.explained_variance_ratio_[2],
              pca.explained_variance_ratio_[3],
              pca.explained_variance_ratio_[4]],
})
diagnostics.to_csv(OUT_DIR / "diagnostics_summary.tsv", sep="\t", index=False)
print(f"✓ Saved: {OUT_DIR / 'diagnostics_summary.tsv'}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"λGC: {lambda_gc:.4f} ({interp})")
print(f"PC1 variance: {pca.explained_variance_ratio_[0]*100:.2f}%")
print(f"PC2 variance: {pca.explained_variance_ratio_[1]*100:.2f}%")
print("\nDone!")