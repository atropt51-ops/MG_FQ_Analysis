"""
Annotation SNP 242471 و بررسی در نمونه‌ها
"""
import pandas as pd
from pathlib import Path

# بارگذاری GFF
gff = pd.read_csv(
    "data/reference/G37.gff3", sep="\t", comment="#", header=None,
    names=["seqid", "source", "type", "start", "end", "score", "strand", "phase", "attributes"]
)

genes = gff[gff["type"] == "gene"].copy()
target_pos = 242471

print("=" * 60)
print(f"Annotation for position {target_pos}")
print("=" * 60)

overlapping = genes[(genes["start"] <= target_pos) & (genes["end"] >= target_pos)]

if len(overlapping) > 0:
    for i, row in overlapping.iterrows():
        print(f"Gene: {row['attributes']}")
        print(f"  Start: {row['start']}, End: {row['end']}, Strand: {row['strand']}")
else:
    genes["distance"] = abs(genes["start"] - target_pos)
    closest = genes.nsmallest(5, "distance")
    print("No gene at exact position. Closest genes:")
    for i, row in closest.iterrows():
        print(f"  {row['attributes']}")
        print(f"    Start: {row['start']}, End: {row['end']}, Distance: {row['distance']} bp")

# بررسی در نمونه‌ها
print("\n" + "=" * 60)
print("Variant 242471 in samples")
print("=" * 60)

mat = pd.read_csv("data/processed/snp_matrix_binary.tsv", sep="\t")
pheno = pd.read_csv("data/processed/phenotype_clean.tsv", sep="\t")

accession_col = "SRR/ERS_Accession "
pheno[accession_col] = pheno[accession_col].astype(str).str.strip()
sample_cols = mat.columns[4:]
pheno = pheno.set_index(accession_col).loc[sample_cols]
y = (pheno["fluoroquinolone_phenotype"] == "R").astype(int).values

# پیدا کردن SNP 242471
mat["variant_id"] = mat["CHROM"] + "_" + mat["POS"].astype(str) + "_" + mat["REF"] + "_" + mat["ALT"]
row = mat[mat["variant_id"] == "NC_000908.2_242471_G_T"]

if len(row) > 0:
    x = row[sample_cols].values[0].astype(float)
    n_var = int((x >= 1).sum())
    a = int(((x >= 1) & (y == 1)).sum())
    b = int(((x >= 1) & (y == 0)).sum())
    c = int(((x == 0) & (y == 1)).sum())
    d = int(((x == 0) & (y == 0)).sum())
    
    print(f"Variant: NC_000908.2_242471_G_T")
    print(f"  Position: {row['POS'].values[0]}")
    print(f"  Total with variant: {n_var}")
    print(f"  Resistant with variant: {a}/{a+c} ({a/(a+c)*100:.1f}%)")
    print(f"  Susceptible with variant: {b}/{b+d} ({b/(b+d)*100:.1f}%)")
    print(f"  Odds Ratio: {(a*d)/(b*c) if b*c > 0 else 'inf'}")
else:
    print("Variant not found")

# بررسی SNPهای اطراف 242471
print("\n" + "=" * 60)
print("All SNPs in 242,000-243,000 region")
print("=" * 60)
region = mat[(mat["POS"] >= 242000) & (mat["POS"] <= 243000)].copy()
print(f"Total SNPs in region: {len(region)}")

for i, row in region.iterrows():
    x = row[sample_cols].values.astype(float)
    n_var = int((x >= 1).sum())
    if n_var >= 5:
        a = int(((x >= 1) & (y == 1)).sum())
        b = int(((x >= 1) & (y == 0)).sum())
        c = int(((x == 0) & (y == 1)).sum())
        d = int(((x == 0) & (y == 0)).sum())
        print(f"  {row['variant_id']}: n_var={n_var}, R={a}/{a+c}, S={b}/{b+d}")