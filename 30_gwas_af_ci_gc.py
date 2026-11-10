"""
30_gwas_af_ci_gc.py (final v2)
Enrich the existing GWAS output with AF, CI, and genomic control.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GWAS_FILE   = "results/gwas_with_pca/gwas_results_pca.tsv"
MATRIX_FILE = "data/processed/snp_matrix_binary.tsv"
OUT_DIR     = "results/gwas_af_ci"
os.makedirs(OUT_DIR, exist_ok=True)


def load_gwas():
    print(f"[INFO] Loading GWAS: {GWAS_FILE}")
    gwas = pd.read_csv(GWAS_FILE, sep="\t")
    print(f"[INFO] Rows: {len(gwas)}, cols: {list(gwas.columns)}")
    return gwas


def load_genotypes_for_af():
    print(f"[INFO] Loading genotypes for AF")
    mat = pd.read_csv(MATRIX_FILE, sep="\t")
    sample_cols = [s for s in mat.columns[4:] if s]
    print(f"[INFO] Matrix: {mat.shape}, samples: {len(sample_cols)}")

    chrom = mat[mat.columns[0]].astype(str).values
    pos   = mat[mat.columns[1]].astype(int).values
    ref   = mat[mat.columns[2]].astype(str).values
    alt   = mat[mat.columns[3]].astype(str).values

    variant = (pd.Series(chrom) + "_" + pd.Series(pos).astype(str)
               + "_" + pd.Series(ref) + "_" + pd.Series(alt)).values

    X = mat[sample_cols].values.astype(np.float64)
    max_val = np.nanmax(X)
    print(f"[INFO] Matrix max: {max_val:.2f}")

    if max_val <= 1.5:
        print(f"[INFO] Coding: BINARY")
        af = np.nanmean(X, axis=1)
        n_alt = np.nansum(X, axis=1)
        n_total = np.sum(~np.isnan(X), axis=1)
    else:
        print(f"[INFO] Coding: DOSAGE")
        n_alt = np.nansum(X, axis=1)
        n_total = np.sum(~np.isnan(X), axis=1)
        af = n_alt / (2 * n_total)

    af = np.clip(af, 0, 1)
    af_df = pd.DataFrame({
        "variant": variant, "af": af,
        "n_alt_alleles": n_alt, "n_samples": n_total,
    }).drop_duplicates("variant", keep="first")

    print(f"[INFO] AF table: {len(af_df)} variants")
    print(f"[INFO] AF range: {af_df['af'].min():.3f} - {af_df['af'].max():.3f}")
    return af_df


def derive_se_from_p(coef, p):
    coef = np.asarray(coef, dtype=float)
    p = np.asarray(p, dtype=float)
    z = norm.isf(np.clip(p / 2, 1e-300, 0.5))
    return np.where(np.abs(z) > 0, np.abs(coef) / z, np.nan)


def lambda_gc(p_values):
    p = np.asarray(p_values, dtype=float)
    p = p[~np.isnan(p) & (p > 0) & (p <= 1)]
    if len(p) == 0: return np.nan
    return float(np.median(chi2.isf(p, df=1)) / 0.4549)


def genomic_control(p_values, lam):
    p = np.asarray(p_values, dtype=float)
    return chi2.sf(chi2.isf(p, df=1) / lam, df=1)


def fdr_bh(p_values):
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    p_clean = np.where(np.isnan(p), 1.0, p)
    order = np.argsort(p_clean)
    ranked = p_clean[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out


def plot_qq(p_values, label, out_path, lam):
    p = np.asarray(p_values, dtype=float)
    p = p[~np.isnan(p) & (p > 0) & (p <= 1)]
    p = np.sort(p)
    expected = -np.log10(np.arange(1, len(p) + 1) / (len(p) + 1))
    observed = -np.log10(p)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(expected, observed, s=8, alpha=0.6, c="steelblue")
    ax.plot([0, expected.max()], [0, expected.max()], "r--", lw=1)
    ax.set_xlabel("Expected -log10(P)")
    ax.set_ylabel("Observed -log10(P)")
    ax.set_title(f"{label}\nLambda = {lam:.3f}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {out_path}")


def plot_manhattan(pos, p_values, out_path, title):
    p = np.asarray(p_values, dtype=float)
    pos_arr = np.asarray(pos, dtype=float)
    valid = ~np.isnan(p) & (p > 0) & (p <= 1)
    p = p[valid]; pos_arr = pos_arr[valid]
    logp = -np.log10(p)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(pos_arr, logp, s=6, alpha=0.6, c="steelblue")
    ax.axhline(-np.log10(5e-8), color="red", ls="--", lw=1, label="5e-8")
    ax.axhline(-np.log10(1e-5), color="orange", ls="--", lw=1, label="1e-5")
    ax.set_xlabel("Position (bp)")
    ax.set_ylabel("-log10(P)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {out_path}")


def main():
    gwas = load_gwas()
    af_df = load_genotypes_for_af()

    merged = gwas.merge(af_df, on="variant", how="left")
    print(f"[INFO] Merged: {len(merged)}, AF missing: {merged['af'].isna().sum()}")

    merged["se"] = derive_se_from_p(merged["coef"].values, merged["p_value"].values)
    merged["or_ci_lo"] = np.exp(np.clip(merged["coef"] - 1.96 * merged["se"], -50, 50))
    merged["or_ci_hi"] = np.exp(np.clip(merged["coef"] + 1.96 * merged["se"], -50, 50))

    lam_raw = lambda_gc(merged["p_value"].values)
    print(f"[INFO] Lambda-GC: {lam_raw:.4f}")
    merged["p_gc"] = genomic_control(merged["p_value"].values, lam_raw)
    lam_gc = lambda_gc(merged["p_gc"].values)
    print(f"[INFO] Lambda-GC after: {lam_gc:.4f}")

    merged["fdr_bh_raw"] = fdr_bh(merged["p_value"].values)
    merged["fdr_bh_gc"]  = fdr_bh(merged["p_gc"].values)

    cols = ["variant", "chrom", "pos", "ref", "alt", "af",
            "n_alt_alleles", "n_samples", "coef", "se",
            "odds_ratio", "or_ci_lo", "or_ci_hi",
            "p_value", "p_gc", "fdr_bh_raw", "fdr_bh_gc"]
    merged = merged[cols].sort_values("p_value").reset_index(drop=True)
    merged.to_csv(os.path.join(OUT_DIR, "gwas_results_enriched.tsv"),
                  sep="\t", index=False)
    print(f"[INFO] Saved: gwas_results_enriched.tsv ({len(merged)} rows)")

    n_fdr_raw = int((merged["fdr_bh_raw"] < 0.05).sum())
    n_fdr_gc  = int((merged["fdr_bh_gc"] < 0.05).sum())

    with open(os.path.join(OUT_DIR, "lambda_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Lambda-GC raw: {lam_raw:.4f}\n")
        f.write(f"Lambda-GC after GC: {lam_gc:.4f}\n")
        f.write(f"FDR<0.05 raw: {n_fdr_raw}\n")
        f.write(f"FDR<0.05 GC: {n_fdr_gc}\n\n")
        f.write("Top 20:\n")
        top = merged.head(20)[["variant", "af", "odds_ratio", "or_ci_lo",
                                "or_ci_hi", "p_value", "p_gc"]]
        f.write(top.to_string(index=False))
        f.write("\n\nparC S83I:\n")
        p1 = merged[merged["variant"].str.contains("242471")]
        f.write(p1[["variant", "af", "odds_ratio", "or_ci_lo",
                    "or_ci_hi", "p_value", "p_gc"]].to_string(index=False))
        f.write("\n")
    print(f"[INFO] Saved: lambda_report.txt")

    plot_qq(merged["p_value"].values, "Raw P",
            os.path.join(OUT_DIR, "qq_plot_raw.png"), lam_raw)
    plot_qq(merged["p_gc"].values, "After GC",
            os.path.join(OUT_DIR, "qq_plot_gc.png"), lam_gc)
    plot_manhattan(merged["pos"].values, merged["p_value"].values,
                   os.path.join(OUT_DIR, "manhattan_raw.png"),
                   f"Manhattan raw (Lambda={lam_raw:.2f})")
    plot_manhattan(merged["pos"].values, merged["p_gc"].values,
                   os.path.join(OUT_DIR, "manhattan_gc.png"),
                   f"Manhattan GC (Lambda={lam_gc:.2f})")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Variants: {len(merged)}")
    print(f"Lambda-GC raw: {lam_raw:.4f}")
    print(f"Lambda-GC after GC: {lam_gc:.4f}")
    print(f"FDR<0.05 raw: {n_fdr_raw}")
    print(f"FDR<0.05 GC: {n_fdr_gc}")
    print("\nTop 10:")
    print(merged.head(10)[["variant", "af", "odds_ratio", "or_ci_lo",
                           "or_ci_hi", "p_value", "p_gc"]].to_string(index=False))
    print("\nparC S83I:")
    p1 = merged[merged["variant"].str.contains("242471")]
    if len(p1):
        print(p1[["variant", "af", "odds_ratio", "or_ci_lo",
                  "or_ci_hi", "p_value", "p_gc"]].to_string(index=False))
    print(f"\n[SAVED] {OUT_DIR}/")


if __name__ == "__main__":
    main()