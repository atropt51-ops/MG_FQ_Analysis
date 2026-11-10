"""
پردازش ماتریس SNP با نام SRR و پر کردن NaN
"""
import pandas as pd
import numpy as np
from pathlib import Path

MATRIX_FILE = Path("results/azzato_2026/snp_matrix_final.tsv")
PHENO_FILE = Path("data/external/azzato_2026_phenotype_linked.tsv")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Loading SNP matrix...")
print("=" * 60)
df = pd.read_csv(MATRIX_FILE, sep="\t", dtype=str)

print(f"Shape: {df.shape}")
print(f"Columns (first 10): {df.columns[:10].tolist()}")

# تبدیل genotype به binary
print("\nConverting genotypes to binary...")
def parse_gt(gt):
    if pd.isna(gt):
        return np.nan
    gt = str(gt).strip()
    if gt in ("0/0", "0|0"):
        return 0
    if gt in ("0/1", "1/0", "0|1", "1|0"):
        return 1
    if gt in ("1/1", "1|1"):
        return 2
    return np.nan

sample_cols = df.columns[4:]
for col in sample_cols:
    df[col] = df[col].apply(parse_gt)

print(f"\nNaN count before fill: {df[sample_cols].isna().sum().sum()}")

# پر کردن NaN با 0
print("Filling NaN with 0...")
df[sample_cols] = df[sample_cols].fillna(0).astype(int)

print(f"NaN count after fill: {df[sample_cols].isna().sum().sum()}")

# ذخیره
out = OUT_DIR / "snp_matrix_binary.tsv"
df.to_csv(out, sep="\t", index=False)
print(f"\n✓ Saved: {out}")
print(f"Final shape: {df.shape}")

# بررسی توزیع
print("\n=== Value distribution (first 3 samples) ===")
for col in sample_cols[:3]:
    print(f"  {col}: {df[col].value_counts().to_dict()}")

# phenotype
print("\n" + "=" * 60)
print("Loading phenotype...")
print("=" * 60)
if PHENO_FILE.exists():
    pheno = pd.read_csv(PHENO_FILE, sep="\t")
    print(f"Pheno shape: {pheno.shape}")
    print(f"FQ-R: {(pheno['fluoroquinolone_phenotype'] == 'R').sum()}")
    print(f"FQ-S: {(pheno['fluoroquinolone_phenotype'] == 'S').sum()}")
    
    sample_cols_vcf = set(df.columns[4:])
    accession_col = "SRR/ERS_Accession "
    if accession_col in pheno.columns:
        pheno[accession_col] = pheno[accession_col].astype(str).str.strip()
        matched = pheno[pheno[accession_col].isin(sample_cols_vcf)]
        print(f"\nMatched samples: {len(matched)} / {len(pheno)}")
    
    pheno.to_csv(OUT_DIR / "phenotype_clean.tsv", sep="\t", index=False)
    print(f"✓ Saved: {OUT_DIR / 'phenotype_clean.tsv'}")
else:
    print(f"WARNING: {PHENO_FILE} not found")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)