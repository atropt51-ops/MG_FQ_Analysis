"""
prep_kmers_small.py (v5)

Line-by-line streaming + random subsampling.
- Keep k-mers in >= MIN_SAMPLES samples
- Randomly subsample to MAX_KMERS if too many
- Memory-safe for 1 GB input

Output: ~50-100 MB
"""

import os
import array
import random

INPUT = "results/kmer_python/kmer_matrix.tsv"
OUTPUT = "results/kmer_python/kmer_matrix_small.tsv"
MIN_SAMPLES = 50       # keep k-mers in >= 50 samples
MAX_KMERS = 50_000     # hard cap after filter
RANDOM_SEED = 42


def main():
    print(f"[INFO] Reading: {INPUT}")
    print(f"[INFO] MIN_SAMPLES = {MIN_SAMPLES}, MAX_KMERS = {MAX_KMERS}")

    # ---- Pass 1: count prevalence ----
    print("[INFO] Pass 1: counting k-mer prevalence (streaming)...")
    n_kmers_total = 0
    n_samples_total = 0

    with open(INPUT, "r") as f:
        header = f.readline().rstrip("\n").split("\t")
        kmer_ids = header[1:]
        n_kmers_total = len(kmer_ids)
        print(f"[INFO] Total k-mers: {n_kmers_total}")

        counts = array.array("I", [0]) * n_kmers_total

        for line_i, line in enumerate(f, 1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) - 1 != n_kmers_total:
                continue
            n_samples_total += 1
            for j, v in enumerate(parts[1:]):
                if v == "1":
                    counts[j] += 1
            if line_i % 40 == 0:
                print(f"[INFO] Pass1: {line_i} samples processed")

    print(f"[INFO] Total samples: {n_samples_total}")

    # ---- Determine keep set ----
    keep_idx = [j for j, c in enumerate(counts) if c >= MIN_SAMPLES]
    print(f"[INFO] k-mers passing >= {MIN_SAMPLES} samples: "
          f"{len(keep_idx)} / {n_kmers_total}")

    # ---- Random subsample if too many ----
    if len(keep_idx) > MAX_KMERS:
        random.seed(RANDOM_SEED)
        keep_idx = sorted(random.sample(keep_idx, MAX_KMERS))
        print(f"[INFO] Randomly subsampled to {len(keep_idx)} k-mers "
              f"(seed={RANDOM_SEED})")
    else:
        print(f"[INFO] No subsampling needed ({len(keep_idx)} <= {MAX_KMERS})")

    del counts

    # ---- Pass 2: write ----
    print("[INFO] Pass 2: writing filtered matrix...")
    with open(INPUT, "r") as f_in, open(OUTPUT, "w") as f_out:
        header = f_in.readline().rstrip("\n").split("\t")
        new_header = [header[0]] + [header[j + 1] for j in keep_idx]
        f_out.write("\t".join(new_header) + "\n")

        for line_i, line in enumerate(f_in, 1):
            parts = line.rstrip("\n").split("\t")
            new_parts = [parts[0]] + [parts[j + 1] for j in keep_idx]
            f_out.write("\t".join(new_parts) + "\n")
            if line_i % 40 == 0:
                print(f"[INFO] Pass2: {line_i} samples written")

    size_mb = os.path.getsize(OUTPUT) / 1024 / 1024
    print(f"[INFO] Done.")
    print(f"[INFO] Output: {OUTPUT} ({size_mb:.1f} MB)")
    print(f"[INFO] k-mers kept: {len(keep_idx)}")


if __name__ == "__main__":
    main()