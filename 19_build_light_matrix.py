"""
ساخت ماتریس سبک (SNP + k-merهای بسیار پرتکرار)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import gc

KMER_FILE = Path("results/kmer_python/kmer_matrix.tsv")
SNP_FILE = Path("data/processed/snp_matrix_binary.tsv")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- مرحله ۱: بارگذاری SNP ---
print("=" * 60)
print("Loading SNP matrix...")
print("=" * 60)
snp = pd.read_csv(SNP_FILE, sep="\t", dtype=str)
snp_samples = snp.columns[4:]
print(f"SNP shape: {snp.shape}")
print(f"SNP samples: {len(snp_samples)}")

# Transpose SNP
snp_T = snp[snp_samples].T
snp_T.index.name = "sample"
print(f"SNP (transposed): {snp_T.shape}")

# --- مرحله ۲: شناسایی k-merهای بسیار پرتکرار ---
print("\n" + "=" * 60)
print("Finding very high-frequency k-mers...")
print("=" * 60)

# header
with open(KMER_FILE) as f:
    header = f.readline().strip().split("\t")
    kmer_cols = header[1:]
    n_samples = sum(1 for _ in open(KMER_FILE)) - 1

print(f"Total k-mer features: {len(kmer_cols)}")
print(f"Total k-mer samples: {n_samples}")

# k-merهای بسیار پرتکرار: >= 80٪ نمونه‌ها
MIN_SAMPLES = int(n_samples * 0.5)  # 50٪
print(f"Minimum samples for k-mer: {MIN_SAMPLES}")

# شمارش
kmer_counts = Counter()
with open(KMER_FILE) as f:
    f.readline()
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        for kmer, val in zip(kmer_cols, parts[1:]):
            if val == "1":
                kmer_counts[kmer] += 1

# فیلتر
valid_kmers = set(k for k, c in kmer_counts.items() if c >= MIN_SAMPLES)
print(f"Valid k-mers (>= {MIN_SAMPLES} samples): {len(valid_kmers)}")
# محدود کردن به ۵۰۰۰۰ k-mer
if len(valid_kmers) > 50000:
    sorted_kmers = sorted(kmer_counts.items(), key=lambda x: -x[1])
    valid_kmers = set(k for k, c in sorted_kmers[:50000])
    print(f"Limited to top 50000 k-mers")
del kmer_counts
gc.collect()

# محدود کردن به ۵۰۰۰۰ k-mer (برای صرفه‌جویی در حافظه)
if len(valid_kmers) > 50000:
    # فقط k-merهای با بالاترین فراوانی
    sorted_kmers = sorted(kmer_counts.items(), key=lambda x: -x[1])
    valid_kmers = set(k for k, c in sorted_kmers[:50000])
    print(f"Limited to top 50000 k-mers")


# --- مرحله ۳: ساخت ماتریس ---
print("\n" + "=" * 60)
print("Building light matrix...")
print("=" * 60)

# لیست ویژگی‌ها
snp_features = snp_T.columns.tolist()
kmer_features = sorted(valid_kmers)
all_features = snp_features + kmer_features

print(f"SNP features: {len(snp_features)}")
print(f"K-mer features: {len(kmer_features)}")
print(f"Total features: {len(all_features)}")

# نمونه‌ها
kmer_samples = []
with open(KMER_FILE) as f:
    f.readline()
    for line in f:
        sample = line.split("\t")[0]
        kmer_samples.append(sample)

common_samples = list(set(snp_T.index) & set(kmer_samples))
print(f"Common samples: {len(common_samples)}")

# ماتریس
n_samples = len(common_samples)
n_features = len(all_features)
print(f"Matrix shape: {n_samples} × {n_features}")

matrix = np.zeros((n_samples, n_features), dtype=np.float32)

# پر کردن SNP
snp_aligned = snp_T.loc[common_samples]
matrix[:, :len(snp_features)] = snp_aligned.values

# پر کردن k-mer
kmer_idx_map = {k: i + len(snp_features) for i, k in enumerate(kmer_features)}
sample_idx_map = {s: i for i, s in enumerate(common_samples)}

print("\nFilling k-mer matrix...")
with open(KMER_FILE) as f:
    f.readline()
    for line in f:
        parts = line.strip().split("\t")
        sample = parts[0]
        if sample not in sample_idx_map:
            continue
        row_idx = sample_idx_map[sample]
        for kmer, val in zip(kmer_cols, parts[1:]):
            if val == "1" and kmer in kmer_idx_map:
                matrix[row_idx, kmer_idx_map[kmer]] = 1

print("Matrix filled")

# ذخیره
print("\n" + "=" * 60)
print("Saving...")
print("=" * 60)

out_df = pd.DataFrame(matrix, index=common_samples, columns=all_features)
out_file = OUT_DIR / "final_matrix_light.tsv"
out_df.to_csv(out_file, sep="\t")
print(f"✓ Saved: {out_file}")
print(f"Final shape: {out_df.shape}")

# phenotype
print("\nSaving phenotype...")
pheno = pd.read_csv("data/processed/phenotype_clean.tsv", sep="\t")
accession_col = "SRR/ERS_Accession "
pheno[accession_col] = pheno[accession_col].astype(str).str.strip()
pheno = pheno.set_index(accession_col)
pheno_subset = pheno.loc[common_samples]
pheno_out = pd.DataFrame({
    "sample": common_samples,
    "resistant": (pheno_subset["fluoroquinolone_phenotype"] == "R").astype(int).values,
})
pheno_out.to_csv(OUT_DIR / "phenotype_light.tsv", sep="\t", index=False)
print(f"✓ Saved: {OUT_DIR / 'phenotype_light.tsv'}")
print(f"Resistant: {pheno_out['resistant'].sum()}, Susceptible: {(pheno_out['resistant']==0).sum()}")

gc.collect()
print("\nDone!")