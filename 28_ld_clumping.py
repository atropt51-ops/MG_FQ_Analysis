"""
28_ld_clumping.py

LD-based clumping of GWAS results to identify independent loci.

Addresses two reviewer concerns:
  1) "989 FDR<0.05 signals are implausible for a clonal pathogen;
      how many INDEPENDENT loci are they?"
  2) "The cluster at 223435-223538 (3 SNPs with identical OR and P)
      is likely a single locus — please resolve."

Method:
  - Compute pairwise LD (r^2) between significant SNPs
  - Clump: for each lead SNP (lowest P in a window), remove SNPs
    within +/- window_kb with r^2 >= r2_threshold
  - Report number of independent loci and their lead SNPs
  - Special diagnostic for the 223xxx cluster

Inputs:
  results/gwas_with_pca/gwas_results_pca.tsv
  data/processed/snp_matrix_binary.tsv
  data/processed/phenotype_clean.tsv

Outputs:
  results/ld_clumping/
    independent_loci.tsv
    cluster_223xxx_report.tsv
    ld_matrix_cluster_223xxx.tsv
    clumping_summary.txt
"""

import os
import re
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
GWAS_FILE   = "results/gwas_with_pca/gwas_results_pca.tsv"
MATRIX_FILE = "data/processed/snp_matrix_binary.tsv"
PHENO_FILE  = "data/processed/phenotype_clean.tsv"
OUT_DIR     = "results/ld_clumping"
os.makedirs(OUT_DIR, exist_ok=True)

FDR_THRESHOLD = 0.05        # clump only FDR-significant SNPs
R2_THRESHOLD  = 0.5         # r^2 cutoff for clumping
WINDOW_KB     = 10          # +/- 10 kb clumping window
CLUSTER_START = 223_000     # 223xxx cluster region
CLUSTER_END   = 224_000

ACCESSION_COL = "SRR/ERS_Accession "
PHENO_COL     = "fluoroquinolone_phenotype"
RESISTANT_VALUE = "R"


def load_gwas():
    print(f"[INFO] Loading GWAS: {GWAS_FILE}")
    gwas = pd.read_csv(GWAS_FILE, sep="\t")
    print(f"[INFO] GWAS rows: {len(gwas)}")
    print(f"[INFO] GWAS columns: {list(gwas.columns)}")
    if "fdr_bh" not in gwas.columns:
        raise ValueError("GWAS file lacks 'fdr_bh' column")
    if "pos" not in gwas.columns:
        raise ValueError("GWAS file lacks 'pos' column")

    # Normalize coordinates and variant IDs
    gwas["pos"] = gwas["pos"].astype(int)
    # Build a canonical variant id if not already present
    if "variant" not in gwas.columns:
        gwas["variant"] = (
            gwas["chrom"].astype(str) + "_" + gwas["pos"].astype(str)
            + "_" + gwas["ref"].astype(str) + "_" + gwas["alt"].astype(str)
        )
    print(f"[INFO] Significant (FDR<{FDR_THRESHOLD}): "
          f"{(gwas['fdr_bh'] < FDR_THRESHOLD).sum()}")
    return gwas


def load_genotypes(variant_ids):
    """
    Load only the SNPs we need from the binary matrix.
    Returns a DataFrame: samples x variants (0/1 coded).
    """
    print(f"[INFO] Loading genotypes for {len(variant_ids)} variants")
    mat = pd.read_csv(MATRIX_FILE, sep="\t")
    print(f"[INFO] Matrix shape: {mat.shape}")

    # Build variant IDs for all SNPs
    if mat.columns[0].upper() in ("CHROM", "CHR", "#CHROM"):
        mat["variant"] = (
            mat[mat.columns[0]].astype(str) + "_" +
            mat[mat.columns[1]].astype(str) + "_" +
            mat[mat.columns[2]].astype(str) + "_" +
            mat[mat.columns[3]].astype(str)
        )
        sample_cols = list(mat.columns[4:-1])
    else:
        mat["variant"] = mat[mat.columns[0]].astype(str)
        sample_cols = list(mat.columns[1:-1])

    subset = mat[mat["variant"].isin(variant_ids)].copy()
    print(f"[INFO] Matched {len(subset)} / {len(variant_ids)} variants")

    if len(subset) == 0:
        raise ValueError("No variant matched between GWAS and matrix")

    geno = subset[sample_cols].astype(np.int8).T
    geno.columns = subset["variant"].values
    print(f"[INFO] Genotype matrix shape: {geno.shape}")
    return geno


def calc_ld_matrix(geno):
    """Pearson r^2 between variants (columns)."""
    print("[INFO] Computing LD (r^2) matrix")
    arr = geno.values.astype(float)
    # Replace any NaNs with column mean (safe; usually none in binary matrix)
    col_means = np.nanmean(arr, axis=0)
    nan_mask = np.isnan(arr)
    if nan_mask.any():
        arr[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    r = np.corrcoef(arr, rowvar=False)
    r2 = r ** 2
    ld = pd.DataFrame(r2, index=geno.columns, columns=geno.columns)
    return ld


def clump(gwas, ld):
    """
    Simple greedy clumping:
      - Order SNPs by ascending P
      - For each SNP not yet assigned:
          assign it as a lead SNP of a new locus
          absorb all SNPs within +/- WINDOW_KB with r^2 >= R2_THRESHOLD
    """
    print(f"[INFO] Clumping with r2>={R2_THRESHOLD}, window=+/-{WINDOW_KB} kb")

    sig = gwas[gwas["fdr_bh"] < FDR_THRESHOLD].copy()
    sig = sig.sort_values("p_value").reset_index(drop=True)
    print(f"[INFO] SNPs to clump: {len(sig)}")

    # Only consider SNPs that are in the LD matrix
    in_ld = set(ld.index)
    sig = sig[sig["variant"].isin(in_ld)].reset_index(drop=True)
    print(f"[INFO] SNPs with genotype data: {len(sig)}")

    assigned = set()
    loci = []

    for i, row in sig.iterrows():
        if row["variant"] in assigned:
            continue
        lead = row["variant"]
        lead_pos = int(row["pos"])
        lead_p = float(row["p_value"])
        lead_or = float(row["odds_ratio"])
        lead_fdr = float(row["fdr_bh"])

        # Candidate neighbours in the LD row
        neighbours = ld.loc[lead]
        window_mask = (
            sig["pos"].between(lead_pos - WINDOW_KB * 1000,
                               lead_pos + WINDOW_KB * 1000)
            & (neighbours.reindex(sig["variant"]).values >= R2_THRESHOLD)
        )
        cluster_variants = set(sig.loc[window_mask, "variant"])
        # Include the lead always
        cluster_variants.add(lead)

        # Skip variants already assigned to an earlier (stronger) locus
        cluster_variants -= assigned
        assigned |= cluster_variants

        loci.append({
            "lead_variant": lead,
            "lead_pos": lead_pos,
            "lead_p": lead_p,
            "lead_or": lead_or,
            "lead_fdr": lead_fdr,
            "n_snps_in_locus": len(cluster_variants),
            "cluster_variants": ";".join(sorted(cluster_variants)),
        })

    loci_df = pd.DataFrame(loci).sort_values("lead_p").reset_index(drop=True)
    loci_df.insert(0, "locus_id", range(1, len(loci_df) + 1))
    print(f"[INFO] Independent loci identified: {len(loci_df)}")
    return loci_df


def report_cluster_223xxx(gwas, geno, ld):
    """Special diagnostic for the 223xxx cluster the reviewer flagged."""
    print(f"\n[INFO] Inspecting 223xxx cluster ({CLUSTER_START}-{CLUSTER_END})")

    in_region = gwas[gwas["pos"].between(CLUSTER_START, CLUSTER_END)].copy()
    in_region = in_region.sort_values("pos")

    # Restrict to variants present in genotype matrix
    in_region = in_region[in_region["variant"].isin(geno.columns)]

    if len(in_region) == 0:
        print("[WARN] No genotype-available variants in 223xxx cluster")
        return None, None

    in_region.to_csv(os.path.join(OUT_DIR, "cluster_223xxx_report.tsv"),
                     sep="\t", index=False)

    # LD sub-matrix
    if len(in_region) > 1:
        sub_ld = ld.loc[in_region["variant"], in_region["variant"]]
        sub_ld.to_csv(os.path.join(OUT_DIR, "ld_matrix_cluster_223xxx.tsv"),
                      sep="\t")
        print(f"[INFO] LD matrix ({len(sub_ld)}x{len(sub_ld)}) saved")

        # Report mean pairwise r^2 among the flagged SNPs
        arr = sub_ld.values.copy()
        np.fill_diagonal(arr, np.nan)
        mean_r2 = np.nanmean(arr)
        print(f"[INFO] Mean pairwise r^2 in 223xxx cluster: {mean_r2:.3f}")

    return in_region, sub_ld


def main():
    gwas = load_gwas()

    # Only genotype-available significant SNPs
    sig_ids = gwas.loc[gwas["fdr_bh"] < FDR_THRESHOLD, "variant"].tolist()
    print(f"[INFO] Significant variant IDs: {len(sig_ids)}")
    geno = load_genotypes(sig_ids)
    ld = calc_ld_matrix(geno)
    loci = clump(gwas, ld)
    loci.to_csv(os.path.join(OUT_DIR, "independent_loci.tsv"),
                sep="\t", index=False)

    # 223xxx cluster diagnostic
    report_cluster_223xxx(gwas, geno, ld)

    # Summary text file
    with open(os.path.join(OUT_DIR, "clumping_summary.txt"), "w") as f:
        f.write("LD clumping summary\n")
        f.write("====================\n\n")
        f.write(f"GWAS file: {GWAS_FILE}\n")
        f.write(f"FDR threshold: {FDR_THRESHOLD}\n")
        f.write(f"r^2 threshold: {R2_THRESHOLD}\n")
        f.write(f"Window: +/- {WINDOW_KB} kb\n\n")
        f.write(f"Significant SNPs (FDR<{FDR_THRESHOLD}): {len(sig_ids)}\n")
        f.write(f"SNPs with genotype data: {len(geno.columns)}\n")
        f.write(f"Independent loci after clumping: {len(loci)}\n\n")
        f.write("Top 20 independent loci:\n")
        f.write(loci.head(20).to_string(index=False))
        f.write("\n")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Significant SNPs (FDR<{FDR_THRESHOLD}): {len(sig_ids)}")
    print(f"Independent loci after clumping: {len(loci)}")
    print("\nTop 10 independent loci:")
    print(loci.head(10)[["locus_id", "lead_variant", "lead_pos",
                         "lead_or", "lead_p", "lead_fdr",
                         "n_snps_in_locus"]].to_string(index=False))
    print(f"\n[SAVED] {OUT_DIR}/")


if __name__ == "__main__":
    main()