"""
GWAS با PCA covariates برای کنترل ساختار جمعیت
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import fisher_exact, chi2
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import gc

MATRIX_FILE = Path("data/processed/snp_matrix_binary.tsv")
PHENO_FILE = Path("data/processed/phenotype_clean.tsv")
PCA_FILE = Path("results/gwas_diagnostics/pca_scores.tsv")
OUT_DIR = Path("results/gwas_with_pca")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Loading data...")
print("=" * 60)

mat = pd.read_csv(MATRIX_FILE, sep="\t")
pheno = pd.read_csv(PHENO_FILE, sep="\t")
pca = pd.read_csv(PCA_FILE, sep="\t", index_col=0)

accession_col = "SRR/ERS_Accession "
pheno[accession_col] = pheno[accession_col].astype(str).str.strip()
sample_cols = mat.columns[4:]
pheno = pheno.set_index(accession_col).loc[sample_cols]
pca = pca.loc[sample_cols]

y = (pheno["fluoroquinolone_phenotype"] == "R").astype(int).values
X = mat[sample_cols].T.values.astype(np.int8)

print(f"Samples: {len(y)}")
print(f"Resistant: {y.sum()}, Susceptible: {(1-y).sum()}")

# PCA covariates (first 5 PCs)
pcs = pca[["PC1", "PC2", "PC3", "PC4", "PC5"]].values

print(f"PCA covariates: {pcs.shape}")

# ============================================================
# GWAS با logistic regression + PCA covariates
# ============================================================
print("\n" + "=" * 60)
print("Running GWAS with PCA covariates...")
print("=" * 60)

records = []
for i, row in mat.iterrows():
    variant = f"{row['CHROM']}_{row['POS']}_{row['REF']}_{row['ALT']}"
    x = row[sample_cols].values.astype(float)
    
    # فقط SNPهایی که واریانت دارند
    n_var = (x >= 1).sum()
    if n_var < 5:
        continue
    
    # Logistic regression با PCA covariates
    X_full = np.column_stack([x, pcs])
    X_full = sm.add_constant(X_full)
    
    try:
        model = sm.Logit(y, X_full).fit(disp=0, maxiter=100)
        coef = model.params[1]  # ضریب SNP
        p_val = model.pvalues[1]
        or_val = np.exp(coef)
        
        records.append({
            "variant": variant,
            "chrom": row["CHROM"],
            "pos": int(row["POS"]),
            "ref": row["REF"],
            "alt": row["ALT"],
            "n_variant": int(n_var),
            "coef": coef,
            "odds_ratio": or_val,
            "p_value": p_val,
        })
    except Exception as e:
        continue
    
    if (i + 1) % 5000 == 0:
        print(f"  Processed {i+1}/{len(mat)} SNPs")

gwas = pd.DataFrame(records)
print(f"\nTested variants: {len(gwas)}")

# FDR
valid = gwas["p_value"].notna()
gwas.loc[valid, "fdr_bh"] = multipletests(gwas.loc[valid, "p_value"], method="fdr_bh")[1]

# آمار
print("\n" + "=" * 60)
print("Results summary:")
print("=" * 60)
print(f"P<0.05: {(gwas['p_value'] < 0.05).sum()}")
print(f"FDR<0.05: {(gwas['fdr_bh'] < 0.05).sum()}")
print(f"FDR<0.001: {(gwas['fdr_bh'] < 0.001).sum()}")

# محاسبه λGC جدید
valid_p = gwas["p_value"].dropna()
valid_p = valid_p[valid_p > 0]
chi2_vals = chi2.isf(valid_p.values, df=1)
lambda_gc = np.median(chi2_vals) / chi2.ppf(0.5, df=1)
print(f"\nλGC (with PCA): {lambda_gc:.4f}")

# ذخیره
gwas.sort_values("p_value").to_csv(OUT_DIR / "gwas_results_pca.tsv", sep="\t", index=False)
print(f"\n✓ Saved: {OUT_DIR / 'gwas_results_pca.tsv'}")

# Top 10
print("\n=== Top 10 SNPs (with PCA) ===")
top = gwas.sort_values("p_value").head(10)
for i, row in top.iterrows():
    print(f"{row['variant']:40s} OR={row['odds_ratio']:.2f} P={row['p_value']:.2e} FDR={row['fdr_bh']:.2e}")

gc.collect()
print("\nDone!")