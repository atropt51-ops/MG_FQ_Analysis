"""
اتصال phenotype Azzato به VCFها
"""
import pandas as pd
from pathlib import Path

EXCEL_PATH = Path("data/external/Azzato_2026_Supplementary_Data_1.xlsx")
VCF_DIR = Path("results/azzato_2026/vcf")
OUT_PATH = Path("data/external/azzato_2026_phenotype_linked.tsv")

print("Loading Excel...")
# header=3: نام ستون‌ها در سطر ۴ (index 3)
df = pd.read_excel(EXCEL_PATH, sheet_name="Sheet1", header=3)

print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print()

# حذف ستون اول خالی
if df.columns[0] == "Unnamed: 0" or pd.isna(df.columns[0]):
    df = df.iloc[:, 1:]
    print(f"After dropping first column: {df.shape}")

print(f"Columns: {df.columns.tolist()}")

# === نام ستون‌ها ===
accession_col = "SRR/ERS_Accession "  # ← دقت: فاصله انتهایی
parc_col = "Genotype_Fluroquinolone _susceptibility_parC_ allele"
gyra_col = "Genotype_Fluroquinolone _susceptibility_gyrA_ allele"

# بررسی وجود ستون‌ها
for col in [accession_col, parc_col, gyra_col]:
    if col not in df.columns:
        print(f"WARNING: '{col}' not found")
        print(f"Available columns with 'SRR': {[c for c in df.columns if 'SRR' in c]}")
        print(f"Available columns with 'parC': {[c for c in df.columns if 'parC' in c]}")
        print(f"Available columns with 'gyrA': {[c for c in df.columns if 'gyrA' in c]}")

# لیست SRR از VCF
vcf_samples = set()
for vcf in VCF_DIR.glob("*.vcf.gz"):
    if ".csi" not in vcf.name:
        srr = vcf.name.replace(".vcf.gz", "")
        vcf_samples.add(srr)
print(f"\nVCF samples: {len(vcf_samples)}")

# فیلتر
if accession_col in df.columns:
    # پاکسازی فاصله‌ها
    df[accession_col] = df[accession_col].astype(str).str.strip()
    df_filtered = df[df[accession_col].isin(vcf_samples)].copy()
    print(f"Matched: {len(df_filtered)}")
else:
    print("ERROR: Cannot match without accession column")
    exit()

# phenotype
def define_fq(parc, gyra):
    if pd.isna(parc) or pd.isna(gyra):
        return None
    parc_str = str(parc).strip()
    gyra_str = str(gyra).strip()
    return "R" if (parc_str != "WT" or gyra_str != "WT") else "S"

if parc_col in df_filtered.columns and gyra_col in df_filtered.columns:
    df_filtered["fluoroquinolone_phenotype"] = df_filtered.apply(
        lambda r: define_fq(r[parc_col], r[gyra_col]),
        axis=1
    )
    
    print("\n=== FQ phenotype ===")
    print(df_filtered["fluoroquinolone_phenotype"].value_counts())
    print("\n=== parC ===")
    print(df_filtered[parc_col].value_counts())
    print("\n=== gyrA ===")
    print(df_filtered[gyra_col].value_counts())
else:
    print("ERROR: parC or gyrA column not found")

# ذخیره
df_filtered.to_csv(OUT_PATH, sep="\t", index=False)
print(f"\n✓ Saved: {OUT_PATH}")
print(f"Final shape: {df_filtered.shape}")