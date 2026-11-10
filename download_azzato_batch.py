"""
Download batch FASTQ from SRA for Azzato 2026
Strategy: download -> gzip -> delete raw
Max space: ~10 GB
"""
import subprocess
import gzip
import shutil
import os
from pathlib import Path

SRR_LIST = Path("data/raw/azzato_2026/srr_list.txt")
OUT_DIR = Path("data/raw/azzato_2026/fastq")
OUT_DIR.mkdir(parents=True, exist_ok=True)

srr_list = [line.strip() for line in SRR_LIST.read_text().strip().split("\n") if line.strip()]
print(f"Total SRR: {len(srr_list)}")
print(f"Already downloaded: {len(list(OUT_DIR.glob('*.fastq.gz'))) // 2}")
print()

downloaded = 0
skipped = 0
errors = []

for i, srr in enumerate(srr_list, 1):
    r1_gz = OUT_DIR / f"{srr}_1.fastq.gz"
    r2_gz = OUT_DIR / f"{srr}_2.fastq.gz"
    
    if r1_gz.exists() and r2_gz.exists():
        skipped += 1
        if i % 10 == 0:
            print(f"[{i}/{len(srr_list)}] Progress: {downloaded} new, {skipped} skipped")
        continue
    
    r1_raw = OUT_DIR / f"{srr}_1.fastq"
    r2_raw = OUT_DIR / f"{srr}_2.fastq"
    
    if r1_raw.exists() and r2_raw.exists():
        print(f"[{i}/{len(srr_list)}] {srr}: gzipping existing...")
        try:
            with open(r1_raw, 'rb') as f_in, gzip.open(r1_gz, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            with open(r2_raw, 'rb') as f_in, gzip.open(r2_gz, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(r1_raw)
            os.remove(r2_raw)
            skipped += 1
            continue
        except Exception as e:
            print(f"  ERROR: {e}")
            errors.append(srr)
            continue
    
    print(f"[{i}/{len(srr_list)}] {srr}: downloading...")
    try:
        cmd = ["fasterq-dump", "--split-files", "--outdir", str(OUT_DIR), srr]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr[:200]}")
            errors.append(srr)
            continue
        
        if r1_raw.exists() and r2_raw.exists():
            print(f"  Gzipping...")
            with open(r1_raw, 'rb') as f_in, gzip.open(r1_gz, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            with open(r2_raw, 'rb') as f_in, gzip.open(r2_gz, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(r1_raw)
            os.remove(r2_raw)
        
        downloaded += 1
        size_mb = (r1_gz.stat().st_size + r2_gz.stat().st_size) / (1024*1024)
        print(f"  OK ({size_mb:.0f} MB) | {downloaded} new, {skipped} skipped")
    
    except subprocess.TimeoutExpired:
        print(f"  ERROR: Timeout")
        errors.append(srr)
    except Exception as e:
        print(f"  ERROR: {e}")
        errors.append(srr)

print()
print("=" * 60)
print(f"Downloaded: {downloaded}")
print(f"Skipped: {skipped}")
print(f"Errors: {len(errors)}")
if errors:
    Path("data/raw/azzato_2026/failed_srr.txt").write_text("\n".join(errors))
    print(f"Failed list: data/raw/azzato_2026/failed_srr.txt")