"""
مقایسه نهایی: SNP تنها vs SNP + k-mer
"""
import pandas as pd

print("=" * 60)
print("Final Comparison: SNP only vs SNP + k-mer")
print("=" * 60)

# SNP only
snp_metrics = pd.read_csv("results/xgboost_azzato/metrics.tsv", sep="\t", index_col=0)
print("\nSNP only (Azzato 2026):")
for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
    if k in snp_metrics.index:
        print(f"  {k:12s}: {snp_metrics.loc[k].values[0]:.4f}")

# Combined
combined_metrics = pd.read_csv("results/xgboost_combined/metrics.tsv", sep="\t", index_col=0)
print("\nSNP + k-mer (Azzato + k-mer):")
for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
    if k in combined_metrics.index:
        print(f"  {k:12s}: {combined_metrics.loc[k].values[0]:.4f}")

print("\n" + "=" * 60)
print("Comparison Table:")
print("=" * 60)
comparison = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
    "SNP only": [
        snp_metrics.loc["accuracy"].values[0],
        snp_metrics.loc["precision"].values[0],
        snp_metrics.loc["recall"].values[0],
        snp_metrics.loc["f1"].values[0],
        snp_metrics.loc["roc_auc"].values[0],
    ],
    "SNP + k-mer": [
        combined_metrics.loc["accuracy"].values[0],
        combined_metrics.loc["precision"].values[0],
        combined_metrics.loc["recall"].values[0],
        combined_metrics.loc["f1"].values[0],
        combined_metrics.loc["roc_auc"].values[0],
    ],
})
print(comparison.to_string(index=False))