"""
Annotation SNPهای برتر parC و gyrA
"""
import pandas as pd
from pathlib import Path

MATRIX_FILE = Path("data/processed/snp_matrix_binary.tsv")
PHENO_FILE = Path("data/processed/phenotype_clean.tsv")

print("Loading data...")
mat = pd.read_csv(MATRIX_FILE, sep="\t")
pheno = pd.read_csv(PHENO_FILE, sep="\t")

accession_col = "SRR/ERS_Accession "
pheno[accession_col] = pheno[accession_col].astype(str).str.strip()
sample_cols = mat.columns[4:]
pheno = pheno.set_index(accession_col).loc[sample_cols]
y = (pheno["fluoroquinolone_phenotype"] == "R").astype(int).values

# SNPهای برتر
parc_variants = [
    "NC_000908.2_273630_T_C",
    "NC_000908.2_273369_T_A",
    "NC_000908.2_270630_G_A",
    "NC_000908.2_273502_A_C",
]

gyra_variants = [
    "NC_000908.2_4739_C_T",
    "NC_000908.2_4833_G_A",
    "NC_000908.2_5096_G_A",
    "NC_000908.2_3963_C_T",
]

def analyze_variant(v):
    row = mat[(mat["CHROM"] + "_" + mat["POS"].astype(str) + "_" + mat["REF"] + "_" + mat["ALT"]) == v]
    if len(row) == 0:
        print(f"{v}: NOT FOUND")
        return
    x = row[sample_cols].values[0].astype(float)
    n_var = int((x >= 1).sum())
    a = int(((x >= 1) & (y == 1)).sum())
    b = int(((x >= 1) & (y == 0)).sum())
    c = int(((x == 0) & (y == 1)).sum())
    d = int(((x == 0) & (y == 0)).sum())
    pos = int(row["POS"].values[0])
    print(f"{v}:")
    print(f"  POS: {pos}")
    print(f"  Variant total: {n_var}")
    print(f"  Resistant with variant: {a}/{a+c} ({a/(a+c)*100:.1f}%)")
    print(f"  Susceptible with variant: {b}/{b+d} ({b/(b+d)*100:.1f}%)")

print("\n" + "=" * 60)
print("parC Top SNPs")
print("=" * 60)
for v in parc_variants:
    analyze_variant(v)

print("\n" + "=" * 60)
print("gyrA Top SNPs")
print("=" * 60)
for v in gyra_variants:
    analyze_variant(v)