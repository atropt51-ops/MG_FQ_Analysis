"""
محاسبه دقیق کدون‌های parC و ارتباط با SNPها
"""
import pandas as pd
from pathlib import Path

# GFF برای parC
gff = pd.read_csv("data/reference/G37.gff3", sep="\t", comment="#", header=None,
                  names=["seqid", "source", "type", "start", "end", "score", "strand", "phase", "attributes"])

parc = gff[gff["attributes"].str.contains("parC", na=False) & (gff["type"] == "gene")]
print("parC gene:")
print(parc[["start", "end", "strand"]].to_string())

start = int(parc["start"].values[0])
end = int(parc["end"].values[0])
strand = parc["strand"].values[0]

print(f"\nparC: {start}-{end} ({end-start+1} bp) on {strand} strand")

# کدون‌های مهم
print("\n=== Codon positions ===")
for codon in [62, 83, 87]:
    codon_start = start + (codon - 1) * 3
    codon_end = codon_start + 2
    print(f"Codon {codon}: positions {codon_start}-{codon_end}")

# SNPهای ناحیه parC
print("\n=== SNPs in parC region (242000-245000) ===")
mat = pd.read_csv("data/processed/snp_matrix_binary.tsv", sep="\t")
mat["variant_id"] = mat["CHROM"] + "_" + mat["POS"].astype(str) + "_" + mat["REF"] + "_" + mat["ALT"]

pheno = pd.read_csv("data/processed/phenotype_clean.tsv", sep="\t")
accession_col = "SRR/ERS_Accession "
pheno[accession_col] = pheno[accession_col].astype(str).str.strip()
sample_cols = mat.columns[4:-1]  # آخرین ستون variant_id است
pheno = pheno.set_index(accession_col).loc[sample_cols]
y = (pheno["fluoroquinolone_phenotype"] == "R").astype(int).values

parc_snps = mat[(mat["POS"] >= 242000) & (mat["POS"] <= 245000)].copy()

for i, row in parc_snps.iterrows():
    pos = int(row["POS"])
    x = row[sample_cols].values.astype(float)
    n_var = int((x >= 1).sum())
    
    if n_var >= 5:
        a = int(((x >= 1) & (y == 1)).sum())
        b = int(((x >= 1) & (y == 0)).sum())
        c = int(((x == 0) & (y == 1)).sum())
        d = int(((x == 0) & (y == 0)).sum())
        
        # محاسبه کدون
        codon = ((pos - start) // 3) + 1
        codon_pos = ((pos - start) % 3) + 1
        
        print(f"{row['variant_id']}: pos={pos}, codon={codon} (pos {codon_pos}), n_var={n_var}, R={a}/{a+c}, S={b}/{b+d}")