"""
ساخت فایل phenotype Fookes 2017
استاندارد: CIP >= 4 -> R (سخت‌گیرانه، مطابق مقاله)
"""
import pandas as pd
from pathlib import Path

# داده‌های Table 1 مقاله Fookes 2017
data = [
    ("M2282",  "Denmark",   1991, 0.004, 0.063, 4,    0.125, 0.25,  0.001, 0, 0, 0),
    ("M2300",  "Denmark",   1991, 0.008, 0.125, 16,   0.125, 0.25,  0.001, 0, 0, 0),
    ("M2341",  "Denmark",   1991, 0.008, 0.25,  2,    0.25,  1,     0.001, 0, 0, 0),
    ("M30",    "UK",        1980, 0.008, 0.125, 8,    0.125, 0.5,   0.001, 0, 0, 0),
    ("M6090",  "France",    1994, 0.004, 0.063, 2,    0.063, 0.125, 0.001, 0, 0, 0),
    ("M6151",  "France",    1994, 0.008, 0.125, 4,    0.125, 0.25,  0.001, 0, 0, 0),
    ("M6257",  "Sweden",    2004, 8,     16,    1,    0.25,  0.5,   2,     1, 0, 0),
    ("M6270",  "Australia", 2004, 8,     16,    4,    0.125, 0.25,  0.5,   1, 0, 0),
    ("M6280",  "Sweden",    1997, 0.004, 0.063, 1,    0.063, 0.125, 0.001, 0, 0, 0),
    ("M6283",  "Japan",     2003, 0.004, 0.063, 2,    0.125, 0.5,   0.001, 0, 0, 0),
    ("M6284",  "Japan",     2003, 0.004, 0.125, 4,    0.125, 0.25,  0.001, 0, 0, 0),
    ("M6285",  "Sweden",    1997, 0.004, 0.125, 1,    0.125, 0.5,   0.001, 0, 0, 0),
    ("M6286",  "Sweden",    2001, 0.002, 0.03,  2,    0.03,  0.25,  0.001, 0, 0, 0),
    ("M6303",  "Norway",    2003, 8,     16,    8,    0.25,  2,     0.5,   1, 0, 0),
    ("M6312",  "France",    1994, 0.004, 0.063, 8,    0.125, 0.25,  0.001, 0, 0, 0),
    ("M6327",  "Denmark",   2005, 0.016, 0.125, 8,    0.25,  0.25,  0.002, 0, 0, 0),
    ("M6328",  "Sweden",    1998, 0.004, 0.125, 2,    0.125, 0.25,  0.001, 0, 0, 0),
    ("M6475",  "Sweden",    2006, 0.008, 0.125, 1,    0.5,   0.5,   0.001, 0, 0, 0),
    ("M6489",  "Sweden",    2007, 16,    16,    16,   16,    1,     1,     1, 0, 1),
    ("M6593",  "Norway",    2008, 16,    16,    2,    0.125, 0.063, 0.5,   1, 0, 0),
    ("M6604",  "Denmark",   2009, 64,    16,    4,    0.25,  0.5,   2,     1, 0, 0),
    ("M6711",  "Australia", 2010, 16,    64,    16,   8,     1,     0.25,  1, 0, 1),
    ("M6713",  "Australia", 2010, 0.063, 0.03,  0.5,  0.125, 0.5,   0.001, 0, 0, 0),
]

cols = ["sample_id", "country", "year", "AZM", "ERY", "CIP", "MXF",
        "DOX", "SOL", "A2058G", "A2059G", "parC_S83I"]

df = pd.DataFrame(data, columns=cols)

# === تعریف phenotype ===
# FQ: CIP >= 4 -> R (سخت‌گیرانه، مطابق Fookes 2017)
df["fluoroquinolone_phenotype"] = df["CIP"].apply(lambda x: "R" if x >= 4 else "S")

# Macrolide: AZM >= 8 -> R
df["macrolide_phenotype"] = df["AZM"].apply(lambda x: "R" if x >= 8 else "S")

# ذخیره
out = Path("data/external/fookes_2017_phenotype.tsv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, sep="\t", index=False)

# === گزارش ===
print(f"✓ Saved: {out}")
print(f"  Total: {len(df)} samples")
print()
print("=== Fluoroquinolone phenotype (CIP >= 4 = R) ===")
print(f"  R: {(df['fluoroquinolone_phenotype']=='R').sum()}")
print(f"  S: {(df['fluoroquinolone_phenotype']=='S').sum()}")
print()
print("=== Macrolide phenotype (AZM >= 8 = R) ===")
print(f"  R: {(df['macrolide_phenotype']=='R').sum()}")
print(f"  S: {(df['macrolide_phenotype']=='S').sum()}")
print()
print("=== FQ-R samples ===")
fq_r = df[df["fluoroquinolone_phenotype"] == "R"]
print(fq_r[["sample_id", "country", "year", "CIP", "MXF", "parC_S83I"]].to_string(index=False))
print()
print("=== FQ-S samples (borderline) ===")
fq_s = df[df["fluoroquinolone_phenotype"] == "S"]
print(fq_s[["sample_id", "country", "year", "CIP", "MXF", "parC_S83I"]].to_string(index=False))