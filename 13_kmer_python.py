"""
شمارش k-mer با پایتون خالص (نسخه بهینه برای حافظه کم)
"""
import gzip
from collections import Counter
from pathlib import Path
import time
import gc

FASTQ_DIR = Path("data/raw/azzato_2026/fastq")
OUT_DIR = Path("results/kmer_python")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# پارامترهای بهینه برای حافظه کم
K = 11                # k-mer کوتاه‌تر برای حافظه کمتر
MIN_COUNT = 2         # حداقل فراوانی
FILTER_EVERY = 100000 # فیلتر مکرر
MAX_KMERS = 500_000   # حداکثر k-mer در حافظه

# لیست نمونه‌ها
samples = []
for r1 in sorted(FASTQ_DIR.glob("*_1.fastq.gz")):
    sample = r1.name.replace("_1.fastq.gz", "")
    r2 = FASTQ_DIR / f"{sample}_2.fastq.gz"
    if r2.exists():
        samples.append((sample, r1, r2))

print(f"Total samples: {len(samples)}")
print(f"K = {K}, MIN_COUNT = {MIN_COUNT}, MAX_KMERS = {MAX_KMERS:,}")
print("=" * 60)

start_time = time.time()
processed = 0
skipped = 0

for i, (sample, r1, r2) in enumerate(samples, 1):
    kmers_file = OUT_DIR / f"{sample}_kmers.tsv"
    
    if kmers_file.exists():
        print(f"[{i}/{len(samples)}] {sample}: already counted")
        skipped += 1
        continue
    
    elapsed = time.time() - start_time
    print(f"[{i}/{len(samples)}] {sample}: counting... (elapsed: {elapsed/60:.1f} min)")
    
    kmers = Counter()
    read_count = 0
    
    try:
        for fastq_file, label in [(r1, "R1"), (r2, "R2")]:
            with gzip.open(fastq_file, "rt") as fh:
                for j, line in enumerate(fh):
                    if j % 4 == 1:
                        seq = line.strip()
                        for pos in range(len(seq) - K + 1):
                            kmer = seq[pos:pos+K]
                            if 'N' not in kmer:
                                kmers[kmer] += 1
                        read_count += 1
                        
                        # فیلتر دوره‌ای
                        if read_count % FILTER_EVERY == 0:
                            # حذف k-merهای با فراوانی ۱
                            to_remove = [k for k, c in kmers.items() if c < MIN_COUNT]
                            for k in to_remove:
                                del kmers[k]
                            
                            # اگر باز هم زیاد بود، فقط top را نگه دار
                            if len(kmers) > MAX_KMERS:
                                sorted_items = sorted(kmers.items(), key=lambda x: -x[1])[:MAX_KMERS]
                                kmers.clear()
                                kmers.update(dict(sorted_items))
                            
                            print(f"  {label}: {read_count//1000}k reads, {len(kmers)} unique kmers")
        
        # ذخیره
        with open(kmers_file, "w") as f:
            for kmer, count in kmers.items():
                if count >= MIN_COUNT:
                    f.write(f"{kmer}\t{count}\n")
        
        processed += 1
        print(f"  ✓ Done ({len(kmers)} kmers, {processed} new, {skipped} skipped)")
    
    except Exception as e:
        print(f"  ERROR: {e}")
    
    finally:
        del kmers
        gc.collect()

print()
print("=" * 60)
print(f"Processed: {processed}")
print(f"Skipped: {skipped}")
print(f"Total time: {(time.time() - start_time)/60:.1f} min")