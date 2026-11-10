"""
ادغام k-mer با SNP بدون خطای dtype
"""
import pandas as pd
import numpy as np
from pathlib import Path

KMER_FILE = Path("results/kmer_python/kmer_matrix.tsv")
SNP_FILE = Path("data/processed/snp_matrix_binary.tsv")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Loading SNP matrix...")
print("=" * 60)

# SNP را بدون dtype خاص بخوان
snp = pd.read_csv(SNP_FILE, sep="\t")
snp_samples = snp.columns[4:]
print(f"SNP shape: {snp.shape}")
print(f"SNP samples: {len(snp_samples)}")

# Transpose SNP
snp_T = snp[snp_samples].T
snp_T.index.name = "sample"
print(f"SNP (transposed): {snp_T.shape}")

# ذخیره SNP
snp_T.to_csv(OUT_DIR / "snp_temp.tsv", sep="\t")

# حالا بدون dtype بخوان
print("\nReloading SNP temp...")
snp_df = pd.read_csv(OUT_DIR / "snp_temp.tsv", sep="\t", index_col=0)
print(f"SNP temp shape: {snp_df.shape}")

# --- ادغام ---
print("\n" + "=" * 60)
print("Merging k-mer with SNP...")
print("=" * 60)

# header k-mer
with open(KMER_FILE) as f:
    header_line = f.readline().strip()
kmer_cols = header_line.split("\t")[1:]
print(f"K-mer features: {len(kmer_cols)}")

# شمارش خطوط
n_samples = sum(1 for _ in open(KMER_FILE)) - 1
print(f"K-mer samples: {n_samples}")

# فایل خروجی
out_file = OUT_DIR / "combined_matrix.tsv"

with open(KMER_FILE) as f_in, open(out_file, "w") as f_out:
    # header
    combined_header = ["sample"] + snp_df.columns.tolist() + kmer_cols
    f_out.write("\t".join(combined_header) + "\n")
    
    # خطوط k-mer
    for i, line in enumerate(f_in):
        parts = line.strip().split("\t")
        sample = parts[0]
        
        if sample in snp_df.index:
            snp_vals = snp_df.loc[sample].astype(str).tolist()
            f_out.write("\t".join([sample] + snp_vals + parts[1:]) + "\n")
        
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{n_samples}] processed")

print(f"\n✓ Saved: {out_file}")
print(f"Final shape: ({n_samples}, {1 + len(snp_df.columns) + len(kmer_cols)})")