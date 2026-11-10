"""
34_shap_direction.py

Compute signed SHAP direction for each variant.

mean |SHAP| tells how important a feature is, but not whether the
variant increases or decreases predicted resistance risk.
"""

import os
import numpy as np
import pandas as pd

SHAP_VALUES = "results/shap_azzato/shap_values.npy"
FEATURES    = "results/shap_azzato/feature_names.txt"
X_MATRIX    = "results/shap_azzato/X.npy"
NONCANON    = "results/noncanonical_table/prioritized_noncanonical_variants.tsv"
OUT_DIR     = "results/shap_azzato"

DIRECTION_THRESHOLD = 1e-3
MIN_SAMPLES_PER_GROUP = 5


def main():
    print("[INFO] Loading SHAP artifacts")
    if not os.path.exists(SHAP_VALUES):
        raise FileNotFoundError(f"Missing: {SHAP_VALUES}")
    if not os.path.exists(FEATURES):
        raise FileNotFoundError(f"Missing: {FEATURES}")

    shap = np.load(SHAP_VALUES)
    print(f"[INFO] shap_values.npy shape: {shap.shape}")

    with open(FEATURES, "r") as f:
        features = [line.strip() for line in f if line.strip()]
    print(f"[INFO] features loaded: {len(features)}")

    if shap.shape[1] != len(features):
        raise ValueError(
            f"SHAP cols ({shap.shape[1]}) != features ({len(features)})"
        )

    X = None
    if os.path.exists(X_MATRIX):
        X = np.load(X_MATRIX)
        print(f"[INFO] X.npy shape: {X.shape}")
        if X.shape[0] != shap.shape[0]:
            print(f"[WARN] X rows ({X.shape[0]}) != SHAP rows ({shap.shape[0]})")
            X = None
    else:
        print("[WARN] X.npy not found — using mean |SHAP| only")

    rows = []
    for j, feat in enumerate(features):
        phi_j = shap[:, j]
        mean_abs_shap = float(np.mean(np.abs(phi_j)))
        mean_shap_signed = float(np.mean(phi_j))

        if X is not None and j < X.shape[1]:
            x_j = X[:, j]
            alt_mask = x_j > 0
            ref_mask = x_j == 0
            n_alt = int(alt_mask.sum())
            n_ref = int(ref_mask.sum())

            if n_alt >= MIN_SAMPLES_PER_GROUP and n_ref >= MIN_SAMPLES_PER_GROUP:
                shap_alt = float(np.mean(phi_j[alt_mask]))
                shap_ref = float(np.mean(phi_j[ref_mask]))
                signed_effect = shap_alt - shap_ref
            else:
                shap_alt = np.nan
                shap_ref = np.nan
                signed_effect = np.nan
        else:
            n_alt = n_ref = np.nan
            shap_alt = shap_ref = np.nan
            signed_effect = np.nan

        if np.isnan(signed_effect):
            direction = "unknown"
        elif abs(signed_effect) < DIRECTION_THRESHOLD:
            direction = "neutral"
        elif signed_effect > 0:
            direction = "risk"
        else:
            direction = "protective"

        rows.append({
            "variant": feat,
            "mean_abs_shap": mean_abs_shap,
            "mean_shap_signed": mean_shap_signed,
            "shap_at_alt": shap_alt,
            "shap_at_ref": shap_ref,
            "signed_effect": signed_effect,
            "n_samples_alt": n_alt,
            "n_samples_ref": n_ref,
            "shap_direction": direction,
        })

    df = pd.DataFrame(rows)
    df["shap_rank"] = df["mean_abs_shap"].rank(ascending=False, method="min")

    out_path = os.path.join(OUT_DIR, "shap_direction.tsv")
    df.to_csv(out_path, sep="\t", index=False)
    print(f"[INFO] Saved: {out_path} ({len(df)} features)")

    print("\n[INFO] Direction counts:")
    print(df["shap_direction"].value_counts().to_string())

    if os.path.exists(NONCANON):
        print(f"\n[INFO] Merging with non-canonical table: {NONCANON}")
        nc = pd.read_csv(NONCANON, sep="\t")
        merged = nc.merge(
            df[["variant", "mean_abs_shap", "mean_shap_signed",
                "shap_at_alt", "shap_at_ref", "signed_effect",
                "shap_direction", "shap_rank"]],
            on="variant", how="left")
        merged_out = "results/noncanonical_table/shap_direction_merged.tsv"
        merged.to_csv(merged_out, sep="\t", index=False)
        print(f"[INFO] Saved: {merged_out}")

        print("\n" + "="*60)
        print("TOP 20 BY |SHAP| (with direction)")
        print("="*60)
        top = merged.sort_values("mean_abs_shap", ascending=False).head(20)
        cols = ["variant", "nearest_gene", "odds_ratio",
                "mean_abs_shap", "signed_effect", "shap_direction"]
        cols = [c for c in cols if c in top.columns]
        print(top[cols].to_string(index=False))

    print(f"\n[SAVED] {OUT_DIR}/")


if __name__ == "__main__":
    main()