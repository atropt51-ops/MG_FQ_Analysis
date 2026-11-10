"""
33_noncanonical_variant_table.py

Build a comprehensive table of prioritized non-canonical variants.

For each top variant (excluding QRDR), report:
  - Coordinates, ref, alt
  - Nearest gene and distance (from GFF)
  - AF in resistant and susceptible groups separately
  - OR, 95% CI, P, FDR (raw and GC-corrected)
  - SHAP value and rank
  - LD locus assignment
  - Stability: how often selected in top-K features per CV fold

Output:
  results/noncanonical_table/
    prioritized_noncanonical_variants.tsv
    summary_by_gene.tsv
"""

import os
import re
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
GWAS_ENRICHED = "results/gwas_af_ci/gwas_results_enriched.tsv"
GFF_FILE      = "data/reference/G37.gff3"
SHAP_FILE     = "results/shap_azzato/shap_importance.tsv"
LOCI_FILE     = "results/ld_clumping/independent_loci.tsv"
MATRIX_FILE   = "data/processed/snp_matrix_binary.tsv"
PHENO_FILE    = "data/processed/phenotype_clean.tsv"
OUT_DIR       = "results/noncanonical_table"
os.makedirs(OUT_DIR, exist_ok=True)

ACCESSION_COL = "SRR/ERS_Accession "
PHENO_COL     = "fluoroquinolone_phenotype"
RESISTANT_VALUE = "R"

# QRDR coordinates (from GFF, verified in script 27)
QRDR_REGIONS = {
    "parC_S83I": (242470, 242472),
    "parC_D87N": (242482, 242484),
    "gyrA_M95I": (5094, 5096),
    "gyrA_83":   (5058, 5060),
    "gyrA_87":   (5070, 5072),
}
PARC_REGION = (242224, 244569)
GYRA_REGION = (4812, 7322)

# Filters
FDR_CUTOFF = 0.05
TOP_N = 50
SHAP_TOP_N = 50


def load_gwas():
    print(f"[INFO] Loading enriched GWAS: {GWAS_ENRICHED}")
    gwas = pd.read_csv(GWAS_ENRICHED, sep="\t")
    print(f"[INFO] GWAS rows: {len(gwas)}, columns: {list(gwas.columns)}")
    return gwas


def load_gff():
    print(f"[INFO] Loading GFF: {GFF_FILE}")
    gff = pd.read_csv(GFF_FILE, sep="\t", comment="#", header=None,
                       names=["seqid", "source", "type", "start", "end",
                              "score", "strand", "phase", "attributes"])
    genes = gff[gff["type"] == "gene"].copy()
    genes["gene_name"] = genes["attributes"].apply(lambda a:
        re.search(r"gene=([^;]+)", a).group(1)
        if re.search(r"gene=([^;]+)", a) else
        (re.search(r"Name=([^;]+)", a).group(1)
         if re.search(r"Name=([^;]+)", a) else "NA"))
    print(f"[INFO] Genes in GFF: {len(genes)}")
    return genes[["start", "end", "strand", "gene_name"]]


def annotate_variants(gwas, genes):
    """For each variant, find nearest gene and distance."""
    print("[INFO] Annotating variants with nearest gene")
    gene_info = []
    for _, row in gwas.iterrows():
        pos = int(row["pos"])
        best_dist = None
        best_gene = "intergenic"
        best_strand = "."
        for _, g in genes.iterrows():
            if g["start"] <= pos <= g["end"]:
                # Inside a gene
                dist = 0
                gene_info.append({
                    "nearest_gene": g["gene_name"],
                    "distance_to_gene": 0,
                    "gene_strand": g["strand"],
                    "region_type": "coding",
                })
                break
            else:
                dist = min(abs(pos - g["start"]), abs(pos - g["end"]))
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_gene = g["gene_name"]
                    best_strand = g["strand"]
        else:
            gene_info.append({
                "nearest_gene": best_gene,
                "distance_to_gene": int(best_dist) if best_dist else -1,
                "gene_strand": best_strand,
                "region_type": "intergenic",
            })
    return pd.DataFrame(gene_info, index=gwas.index)


def is_qrdr(pos):
    for (lo, hi) in QRDR_REGIONS.values():
        if lo <= pos <= hi:
            return True
    return False


def is_near_qrdr_gene(pos, window=500):
    """Flag variants within ±500bp of parC or gyrA genes."""
    if PARC_REGION[0] - window <= pos <= PARC_REGION[1] + window:
        return True
    if GYRA_REGION[0] - window <= pos <= GYRA_REGION[1] + window:
        return True
    return False


def add_af_by_group(gwas):
    """Compute AF separately in resistant and susceptible groups."""
    print("[INFO] Computing AF in resistant vs susceptible groups")
    mat = pd.read_csv(MATRIX_FILE, sep="\t")
    sample_cols = list(mat.columns[4:])
    pheno = pd.read_csv(PHENO_FILE, sep="\t")
    pheno[ACCESSION_COL] = pheno[ACCESSION_COL].astype(str).str.strip()
    pheno = pheno.set_index(ACCESSION_COL).reindex(sample_cols)
    y = (pheno[PHENO_COL] == RESISTANT_VALUE).astype(int).values

    variant = (
        mat[mat.columns[0]].astype(str) + "_" +
        mat[mat.columns[1]].astype(str) + "_" +
        mat[mat.columns[2]].astype(str) + "_" +
        mat[mat.columns[3]].astype(str)
    ).values

    X = mat[sample_cols].values.astype(np.float64)
    n_alt = X.sum(axis=1)
    n_total = X.shape[1]

    if np.nanmax(X) <= 1.5:
        af_all = X.mean(axis=1)
    else:
        af_all = n_alt / (2 * n_total)

    # Group-specific AF
    R_mask = (y == 1)
    S_mask = (y == 0)
    X_R = X[:, R_mask]
    X_S = X[:, S_mask]
    if np.nanmax(X) <= 1.5:
        af_R = X_R.mean(axis=1)
        af_S = X_S.mean(axis=1)
    else:
        af_R = X_R.sum(axis=1) / (2 * X_R.shape[1])
        af_S = X_S.sum(axis=1) / (2 * X_S.shape[1])

    af_df = pd.DataFrame({
        "variant": variant,
        "af_resistant": af_R,
        "af_susceptible": af_S,
        "af_difference": af_R - af_S,
    }).drop_duplicates("variant").reset_index(drop=True)
    print(f"[INFO] AF table: {len(af_df)} variants")
    return af_df


def load_shap():
    print(f"[INFO] Loading SHAP: {SHAP_FILE}")
    if not os.path.exists(SHAP_FILE):
        print("[WARN] SHAP file not found — SHAP columns will be missing")
        return None
    shap = pd.read_csv(SHAP_FILE, sep="\t")
    print(f"[INFO] SHAP columns: {list(shap.columns)}")
    return shap


def load_loci():
    print(f"[INFO] Loading LD loci: {LOCI_FILE}")
    if not os.path.exists(LOCI_FILE):
        return None
    loci = pd.read_csv(LOCI_FILE, sep="\t")
    # Explode cluster_variants
    exploded = []
    for _, row in loci.iterrows():
        variants = str(row["cluster_variants"]).split(";")
        for v in variants:
            exploded.append({
                "variant": v.strip(),
                "locus_id": row["locus_id"],
                "locus_lead": row["lead_variant"],
                "n_snps_in_locus": row["n_snps_in_locus"],
            })
    return pd.DataFrame(exploded).drop_duplicates("variant")


def compute_fold_stability():
    """
    Read per-fold feature selection if available.
    If not, compute a simple proxy: how many folds selected each variant
    from the 32 script folds — but that file lacks feature IDs.
    We use SHAP rank stability instead if no fold selection file exists.
    """
    print("[INFO] Fold stability will use SHAP rank if available")
    return None


def main():
    gwas = load_gwas()
    genes = load_gff()

    # Annotate
    anno = annotate_variants(gwas, genes)
    gwas = pd.concat([gwas, anno], axis=1)

    # AF by group
    af_by_group = add_af_by_group(gwas)
    gwas = gwas.merge(af_by_group, on="variant", how="left")

    # QRDR exclusion
    gwas["is_qrdr"] = gwas["pos"].apply(is_qrdr)
    gwas["near_qrdr_gene"] = gwas["pos"].apply(is_near_qrdr_gene)

    # SHAP
    shap = load_shap()
    if shap is not None:
        # Try common column names
        if "feature" in shap.columns:
            shap_col = "feature"
        elif "variant" in shap.columns:
            shap_col = "variant"
        else:
            shap_col = shap.columns[0]
        shap_value_col = [c for c in shap.columns if "mean" in c.lower() or "shap" in c.lower()][0]
        shap = shap.rename(columns={shap_col: "variant",
                                     shap_value_col: "shap_value"})
        shap["shap_rank"] = shap["shap_value"].rank(ascending=False, method="min")
        gwas = gwas.merge(shap[["variant", "shap_value", "shap_rank"]],
                           on="variant", how="left")

    # LD loci
    loci = load_loci()
    if loci is not None:
        gwas = gwas.merge(loci, on="variant", how="left")

    # Build prioritized non-canonical list
    print("\n[INFO] Selecting prioritized non-canonical variants")
    cand = gwas[(~gwas["is_qrdr"]) & (~gwas["near_qrdr_gene"])].copy()
    print(f"[INFO] Candidates (excluding QRDR ±500bp): {len(cand)}")

    # Sort by FDR, then by SHAP if present
    cand = cand.sort_values("p_value").reset_index(drop=True)

    # Top N by FDR
    top = cand.head(TOP_N).copy()

    # Also top SHAP (if available) that aren't QRDR
    if "shap_value" in cand.columns:
        top_shap = cand.dropna(subset=["shap_value"]).sort_values(
            "shap_value", ascending=False).head(SHAP_TOP_N)
        extra = top_shap[~top_shap["variant"].isin(top["variant"])]
        top = pd.concat([top, extra]).drop_duplicates("variant")

    # Order columns for final table
    cols_order = [
        "variant", "chrom", "pos", "ref", "alt",
        "nearest_gene", "distance_to_gene", "region_type",
        "af_resistant", "af_susceptible", "af_difference",
        "odds_ratio", "or_ci_lo", "or_ci_hi",
        "p_value", "p_gc", "fdr_bh_raw", "fdr_bh_gc",
        "shap_value", "shap_rank",
        "locus_id", "locus_lead", "n_snps_in_locus",
        "is_qrdr", "near_qrdr_gene",
    ]
    cols_present = [c for c in cols_order if c in top.columns]
    top = top[cols_present]

    top.to_csv(os.path.join(OUT_DIR, "prioritized_noncanonical_variants.tsv"),
               sep="\t", index=False)
    print(f"[INFO] Saved: prioritized_noncanonical_variants.tsv ({len(top)} rows)")

    # Summary by gene
    if "nearest_gene" in top.columns:
        gene_summary = top.groupby("nearest_gene").agg(
            n_variants=("variant", "count"),
            min_p=("p_value", "min"),
            min_fdr=("fdr_bh_raw", "min"),
            mean_af_R=("af_resistant", "mean"),
            mean_af_S=("af_susceptible", "mean"),
        ).sort_values("n_variants", ascending=False)
        gene_summary.to_csv(os.path.join(OUT_DIR, "summary_by_gene.tsv"),
                             sep="\t")
        print(f"[INFO] Saved: summary_by_gene.tsv ({len(gene_summary)} genes)")

    # Preview
    print("\n" + "="*60)
    print("TOP 20 PRIORITIZED NON-CANONICAL VARIANTS")
    print("="*60)
    preview_cols = ["variant", "pos", "nearest_gene", "af_resistant",
                    "af_susceptible", "odds_ratio", "p_value", "fdr_bh_raw"]
    preview_cols = [c for c in preview_cols if c in top.columns]
    print(top[preview_cols].head(20).to_string(index=False))
    print(f"\n[SAVED] {OUT_DIR}/")


if __name__ == "__main__":
    main()