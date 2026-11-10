"""
GWAS: Fisher exact test + FDR برای Azzato 2026
"""
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from pathlib import Path
import time

MATRIX_FILE = Path("data/processed/snp_matrix_binary.tsv")
PHENO_FILE = Path("data/processed/phenotype_clean.tsv")
OUT_DIR = Path("results/gwas_azzato")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Loading data...")
print("=" * 60)
mat = pd.read_csv(MATRIX_FILE, sep="\t")
pheno = pd.read_csv(PHENO_FILE, sep="\t")

accession_col = "SRR/ERS_Accession "
pheno[accession_col] = pheno[accession_col].astype(str).str.strip()
sample_cols = mat.columns[4:]

# هم‌ترازی
pheno = pheno.set_index(accession_col).loc[sample_cols]
y = (pheno["fluoroquinolone_phenotype"] == "R").astype(int).values

print(f"Samples: {len(y)}")
print(f"Resistant: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
print(f"Susceptible: {(1-y).sum()} ({(1-y).sum()/len(y)*100:.1f}%)")

# Fisher test
print("\n" + "=" * 60)
print("Running Fisher exact tests...")
print("=" * 60)
start = time.time()
records = []

for i, row in mat.iterrows():
    variant = f"{row['CHROM']}_{row['POS']}_{row['REF']}_{row['ALT']}"
    x = row[sample_cols].values.astype(float)
    
    # فقط SNPهایی که حداقل ۵ نمونه واریانت دارند
    n_variant = (x >= 1).sum()
    if n_variant < 5:
        continue
    
    # جدول 2x2: variant present (≥1) vs absent (0)
    a = int(((x >= 1) & (y == 1)).sum())  # present + resistant
    b = int(((x >= 1) & (y == 0)).sum())  # present + susceptible
    c = int(((x == 0) & (y == 1)).sum())  # absent + resistant
    d = int(((x == 0) & (y == 0)).sum())  # absent + susceptible
    
    if a + b == 0 or c + d == 0 or a + c == 0 or b + d == 0:
        continue
    
    or_val, p_val = fisher_exact([[a, b], [c, d]])
    
    records.append({
        "variant": variant,
        "chrom": row['CHROM'],
        "pos": int(row['POS']),
        "ref": row['REF'],
        "alt": row['ALT'],
        "a_present_res": a,
        "b_present_sus": b,
        "c_absent_res": c,
        "d_absent_sus": d,
        "n_variant_total": a + b,
        "odds_ratio": or_val,
        "p_value": p_val,
    })
    
    if (i + 1) % 5000 == 0:
        elapsed = time.time() - start
        print(f"  Processed {i+1}/{len(mat)} SNPs (elapsed: {elapsed/60:.1f} min)")

gwas = pd.DataFrame(records)
print(f"\nTested variants: {len(gwas)}")

# FDR
valid = gwas["p_value"].notna()
gwas.loc[valid, "fdr_bh"] = multipletests(gwas.loc[valid, "p_value"], method="fdr_bh")[1]

print("\n" + "=" * 60)
print("Results summary:")
print("=" * 60)
print(f"P<0.05:       {(gwas['p_value'] < 0.05).sum()}")
print(f"P<0.01:       {(gwas['p_value'] < 0.01).sum()}")
print(f"P<0.001:      {(gwas['p_value'] < 0.001).sum()}")
print(f"FDR<0.05:     {(gwas['fdr_bh'] < 0.05).sum()}")
print(f"FDR<0.01:     {(gwas['fdr_bh'] < 0.01).sum()}")
print(f"FDR<0.001:    {(gwas['fdr_bh'] < 0.001).sum()}")

# ذخیره
gwas_sorted = gwas.sort_values("p_value")
out_file = OUT_DIR / "gwas_results.tsv"
gwas_sorted.to_csv(out_file, sep="\t", index=False)
print(f"\n✓ Saved: {out_file}")

# Top 20
print("\n" + "=" * 60)
print("Top 20 significant variants:")
print("=" * 60)
top20 = gwas_sorted.head(20)
print(top20[["variant", "n_variant_total", "odds_ratio", "p_value", "fdr_bh"]].to_string(index=False))

# پارC/gyrA
print("\n" + "=" * 60)
print("Checking known resistance loci (parC, gyrA):")
print("=" * 60)
known_loci = gwas_sorted[
    gwas_sorted["variant"].str.contains("parC|gyrA", case=False, na=False)
]
if len(known_loci) > 0:
    print(known_loci[["variant", "odds_ratio", "p_value", "fdr_bh"]].head(20).to_string(index=False))

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)