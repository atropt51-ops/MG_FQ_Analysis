"""
Align FASTQ with G37 and call variants
No external dependencies required - uses only standard library
"""
import subprocess
from pathlib import Path
import time

FASTQ_DIR = Path("data/raw/azzato_2026/fastq")
BAM_DIR = Path("results/azzato_2026/bam")
VCF_DIR = Path("results/azzato_2026/vcf")
REF = "data/reference/G37.fasta"

BAM_DIR.mkdir(parents=True, exist_ok=True)
VCF_DIR.mkdir(parents=True, exist_ok=True)

if not Path(REF).exists():
    print(f"ERROR: {REF} not found")
    exit()

if not Path(f"{REF}.bwt").exists():
    print("Indexing reference with bwa...")
    subprocess.run(["bwa", "index", REF], check=True)
    print("✓ Index created")

samples = []
for r1 in sorted(FASTQ_DIR.glob("*_1.fastq.gz")):
    sample = r1.name.replace("_1.fastq.gz", "")
    r2 = FASTQ_DIR / f"{sample}_2.fastq.gz"
    if r2.exists():
        samples.append((sample, r1, r2))

print(f"Found {len(samples)} samples")
print("=" * 60)

start_time = time.time()
processed = 0
skipped = 0
errors = []

for i, (sample, r1, r2) in enumerate(samples, 1):
    vcf = VCF_DIR / f"{sample}.vcf.gz"
    
    if vcf.exists():
        skipped += 1
        if i % 10 == 0:
            print(f"[{i}/{len(samples)}] {sample}: skipped")
        continue
    
    bam = BAM_DIR / f"{sample}.sorted.bam"
    elapsed = time.time() - start_time
    print(f"[{i}/{len(samples)}] {sample}: aligning... (elapsed: {elapsed/60:.0f} min)")
    
    try:
        cmd = f'bwa mem -t 8 "{REF}" "{r1}" "{r2}" 2>/dev/null | samtools sort -@ 8 -o "{bam}" -'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            print(f"  BWA ERROR: {result.stderr[:300]}")
            errors.append(sample)
            continue
        
        subprocess.run(["samtools", "index", str(bam)], check=True)
        
        cmd = f'bcftools mpileup -f "{REF}" -q 20 -Q 20 -a AD,DP "{bam}" 2>/dev/null | bcftools call -mv -Oz -o "{vcf}" 2>/dev/null'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            print(f"  BCFTOOLS ERROR: {result.stderr[:300]}")
            errors.append(sample)
            continue
        
        subprocess.run(["bcftools", "index", str(vcf)], check=True)
        bam.unlink()
        bai = BAM_DIR / f"{sample}.sorted.bam.bai"
        if bai.exists():
            bai.unlink()
        
        processed += 1
        print(f"  ✓ Done ({processed} new, {skipped} skipped)")
    
    except subprocess.TimeoutExpired:
        print(f"  ERROR: Timeout")
        errors.append(sample)
    except Exception as e:
        print(f"  ERROR: {e}")
        errors.append(sample)

elapsed = time.time() - start_time
print()
print("=" * 60)
print(f"Processed: {processed}")
print(f"Skipped: {skipped}")
print(f"Errors: {len(errors)}")
print(f"Total time: {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")
if errors:
    Path("results/azzato_2026/failed_alignment.txt").write_text("\n".join(errors))
    print(f"Failed: {errors}")
