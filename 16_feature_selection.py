"""
کاهش ویژگی با SelectKBest (نسخه اصلاح‌شده برای خطای dtype)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_selection import SelectKBest, f_classif
import gc

COMBINED_FILE = Path("data/processed/combined_matrix.tsv")
PHENO_FILE = Path("data/processed/phenotype_clean.tsv")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Loading combined matrix...")
print("=" * 60)

# --- روش ۱: خواندن با dtype برای همه ستون‌ها به جز ایندکس ---
# نام ستون‌ها را از header می‌خوانیم
with open(COMBINED_FILE) as f:
    header = f.readline().strip().split("\t")
feature_names = header[1:]
print(f"Total features: {len(feature_names)}")

# خواندن بدون index_col (ایندکس به عنوان ستون اول)
combined = pd.read_csv(
    COMBINED_FILE,
    sep="\t",
    dtype={col: np.float32 for col in feature_names},
)
print(f"Shape: {combined.shape}")

# ستون اول = نام نمونه
sample_col = combined.columns[0]
print(f"First column: {sample_col}")
print(f"Sample names: {combined[sample_col].head().tolist()}")

# تنظیم ایندکس
combined = combined.set_index(sample_col)
print(f"After set_index: {combined.shape}")
print(f"Index: {combined.index[:3].tolist()}")

# phenotype
print("\nLoading phenotype...")
pheno = pd.read_csv(PHENO_FILE, sep="\t")
accession_col = "SRR/ERS_Accession "
pheno[accession_col] = pheno[accession_col].astype(str).str.strip()
pheno = pheno.set_index(accession_col)
y = (pheno.loc[combined.index, "fluoroquinolone_phenotype"] == "R").astype(int).values

print(f"Samples: {len(y)}")
print(f"Resistant: {y.sum()}, Susceptible: {(1-y).sum()}")

# --- SelectKBest ---
print("\n" + "=" * 60)
print("Selecting top 10000 features...")
print("=" * 60)

X = combined.values.astype(np.float32)
selector = SelectKBest(f_classif, k=min(10000, X.shape[1]))
X_filtered = selector.fit_transform(X, y)
selected_idx = selector.get_support(indices=True)
selected_features = [feature_names[i] for i in selected_idx]

print(f"After filter: {X_filtered.shape}")

# ذخیره
out_df = pd.DataFrame(X_filtered, index=combined.index, columns=selected_features)
out_file = OUT_DIR / "final_matrix.tsv"
out_df.to_csv(out_file, sep="\t")
print(f"\n✓ Saved: {out_file}")
print(f"Final shape: {out_df.shape}")

# phenotype
pheno_out = pd.DataFrame({"sample": combined.index, "resistant": y})
pheno_out.to_csv(OUT_DIR / "final_phenotype.tsv", sep="\t", index=False)
print(f"✓ Saved: {OUT_DIR / 'final_phenotype.tsv'}")

# آزادسازی
del combined, X, X_filtered, out_df
gc.collect()
print("\nDone!")