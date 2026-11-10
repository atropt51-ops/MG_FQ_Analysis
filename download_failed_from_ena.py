"""
Download failed SRR from ENA instead of NCBI
"""
import urllib.request
import os
from pathlib import Path
import pandas as pd

OUT_DIR = Path("data/raw/azzato_2026/fastq")
OUT_DIR.mkdir(parents=True, exist_ok=True)

failed_file = Path("data/raw/azzato_2026/failed_srr.txt")
if not failed_file.exists():
    print(f"ERROR: {failed_file} not found")
    exit()

srr_list = [l.strip() for l in failed_file.read_text().strip().split("\n") if l.strip()]
print(f"Failed SRR to retry from ENA: {len(srr_list)}")
print()

downloaded = 0
skipped = 0
errors = []

for i, srr in enumerate(srr_list, 1):
    r1_gz = OUT_DIR / f"{srr}_1.fastq.gz"
    r2_gz = OUT_DIR / f"{srr}_2.fastq.gz"
    
    if r1_gz.exists() and r2_gz.exists():
        print(f"[{i}/{len(srr_list)}] {srr}: already exists")
        skipped += 1
        continue
    
    print(f"[{i}/{len(srr_list)}] {srr}: fetching from ENA...")
    api_url = f"https://www.ebi.ac.uk/ena/portal/api/filereport?accession={srr}&result=read_run&fields=fastq_ftp&format=tsv"
    
    try:
        df = pd.read_csv(api_url, sep="\t")
        if df.empty or "fastq_ftp" not in df.columns:
            print(f"  ERROR: no fastq_ftp for {srr}")
            errors.append(srr)
            continue
        
        ftp_str = df["fastq_ftp"].iloc[0]
        urls = ftp_str.split(";")
        
        success = True
        for url in urls:
            https_url = "https://" + url
            filename = url.split("/")[-1]
            
            if "_1.fastq.gz" in filename:
                r_part = "1"
            elif "_2.fastq.gz" in filename:
                r_part = "2"
            else:
                continue
            
            new_filename = f"{srr}_{r_part}.fastq.gz"
            out_path = OUT_DIR / new_filename
            
            if out_path.exists():
                continue
            
            print(f"  Downloading {new_filename}...")
            urllib.request.urlretrieve(https_url, out_path)
            size_mb = out_path.stat().st_size / (1024*1024)
            print(f"  OK ({size_mb:.0f} MB)")
        
        if success:
            downloaded += 1
            print(f"  ✓ Done ({downloaded} new, {skipped} skipped)")
    except Exception as e:
        print(f"  ERROR: {e}")
        errors.append(srr)

print()
print("=" * 60)
print(f"Downloaded: {downloaded}")
print(f"Skipped: {skipped}")
print(f"Errors: {len(errors)}")
if errors:
    Path("data/raw/azzato_2026/ena_failed_srr.txt").write_text("\n".join(errors))
    print(f"Still failed: {errors}")