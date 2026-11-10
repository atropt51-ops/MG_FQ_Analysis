"""
Download all FASTQ files for PRJEB5172 (Fookes 2017) from ENA
"""
import pandas as pd
import urllib.request
from pathlib import Path

OUT_DIR = Path("data/raw/prjeb5172_fastq")
OUT_DIR.mkdir(parents=True, exist_ok=True)

API_URL = (
    "https://www.ebi.ac.uk/ena/portal/api/filereport"
    "?accession=PRJEB5172"
    "&result=read_run"
    "&fields=run_accession,fastq_ftp"
    "&format=tsv"
)

print("Fetching file list from ENA...")
df = pd.read_csv(API_URL, sep="\t")
print(f"Number of runs: {len(df)}")
print(df.head().to_string())

for idx, row in df.iterrows():
    run = row["run_accession"]
    ftp_str = row["fastq_ftp"]

    if pd.isna(ftp_str):
        print(f"[{idx+1}] {run}: no FASTQ - skipped")
        continue

    urls = ftp_str.split(";")
    for url in urls:
        url = url.strip()
        if not url:
            continue

        https_url = "https://" + url
        filename = url.split("/")[-1]
        out_path = OUT_DIR / filename

        if out_path.exists():
            print(f"[{idx+1}] {filename}: already downloaded")
            continue

        print(f"[{idx+1}] Downloading {filename}...")
        try:
            urllib.request.urlretrieve(https_url, out_path)
            size_mb = out_path.stat().st_size / (1024 * 1024)
            print(f"      OK {size_mb:.1f} MB")
        except Exception as e:
            print(f"      ERROR: {e}")

print("\n=== Download finished ===")
print(f"Files in: {OUT_DIR.absolute()}")
