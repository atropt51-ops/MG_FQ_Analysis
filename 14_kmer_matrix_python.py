"""
ساخت ماتریس presence/absence k-mer
نسخه بهینه برای حافظه کم (۱.۳ GB آزاد)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import gc

KMER_DIR = Path("results/kmer_python")
OUT_DIR = Path("results/kmer_python")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# لیست نمونه‌ها
samples = []
for f in sorted(KMER_DIR.glob("*_kmers.tsv")):
    sample = f.name.replace("_kmers.tsv", "")
    samples.append(sample)

print(f"Total samples: {len(samples)}")

# حداقل تعداد نمونه برای حفظ k-mer
MIN_SAMPLES = 10

# --- مرحله ۱: شمارش k-merهای معتبر (بدون ذخیره همه در حافظه) ---
print("\nStep 1: Finding valid k-mers (in >= %d samples)..." % MIN_SAMPLES)

# برای صرفه‌جویی در حافظه، از Counter با محدودیت استفاده می‌کنیم
from collections import Counter

kmer_counts = Counter()
BATCH_SIZE = 20

for i, sample in enumerate(samples, 1):
    kmers_file = KMER_DIR / f"{sample}_kmers.tsv"
    with open(kmers_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                kmer = parts[0]
                kmer_counts[kmer] += 1
    
    if i % BATCH_SIZE == 0:
        print(f"  [{i}/{len(samples)}] {len(kmer_counts)} unique k-mers")
        gc.collect()

print(f"\nTotal unique k-mers: {len(kmer_counts)}")

# فیلتر k-merهای معتبر
valid_kmers = set(k for k, c in kmer_counts.items() if c >= MIN_SAMPLES)
print(f"Valid k-mers (>= {MIN_SAMPLES} samples): {len(valid_kmers)}")

# آزادسازی حافظه
del kmer_counts
gc.collect()

# --- مرحله ۲: ساخت ماتریس ---
print("\nStep 2: Building matrix...")
kmer_list = sorted(valid_kmers)
n_kmers = len(kmer_list)
n_samples = len(samples)

print(f"Matrix shape: {n_samples} samples × {n_kmers} kmers")

# ساخت ماتریس
matrix = np.zeros((n_samples, n_kmers), dtype=np.int8)
kmer_to_idx = {k: i for i, k in enumerate(kmer_list)}

for i, sample in enumerate(samples, 1):
    kmers_file = KMER_DIR / f"{sample}_kmers.tsv"
    with open(kmers_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                kmer = parts[0]
                if kmer in kmer_to_idx:
                    matrix[i-1, kmer_to_idx[kmer]] = 1
    
    if i % 20 == 0:
        print(f"  [{i}/{len(samples)}] filled")

# ذخیره
out_df = pd.DataFrame(matrix, index=samples, columns=kmer_list)
out_file = OUT_DIR / "kmer_matrix.tsv"
out_df.to_csv(out_file, sep="\t")
print(f"\n✓ Saved: {out_file}")
print(f"Final shape: {out_df.shape}")
print(f"Matrix density: {matrix.sum() / matrix.size * 100:.2f}%")