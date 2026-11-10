"""
ساخت شکل‌های نهایی مقاله
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix

OUT_DIR = Path("results/figures_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# بارگذاری
metrics = pd.read_csv("results/xgboost_azzato/metrics.tsv", sep="\t", index_col=0)
preds = pd.read_csv("results/xgboost_azzato/predictions.tsv", sep="\t")
snp_metrics = pd.read_csv("results/xgboost_azzato/metrics.tsv", sep="\t", index_col=0)
combined_metrics = pd.read_csv("results/xgboost_combined/metrics.tsv", sep="\t", index_col=0)

print("Generating figures...")

# --- Figure 1: ROC Curve ---
y_true = preds["y_true"].values
y_proba = preds["y_proba"].values
fpr, tpr, _ = roc_curve(y_true, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curve — XGBoost + SMOTE (LOO)", fontsize=14)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "figure_roc.png", dpi=150)
plt.close()
print("  Figure 1: ROC curve ✓")

# --- Figure 2: Confusion Matrix ---
cm = confusion_matrix(y_true, preds["y_pred"].values)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Susceptible", "Resistant"])
ax.set_yticklabels(["Susceptible", "Resistant"])
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("Actual", fontsize=12)
ax.set_title("Confusion Matrix — XGBoost + SMOTE", fontsize=14)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=16)
plt.colorbar(im)
plt.tight_layout()
plt.savefig(OUT_DIR / "figure_confusion_matrix.png", dpi=150)
plt.close()
print("  Figure 2: Confusion matrix ✓")

# --- Figure 3: Metrics Bar ---
metric_names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
metric_vals = [
    metrics.loc["accuracy"].values[0],
    metrics.loc["precision"].values[0],
    metrics.loc["recall"].values[0],
    metrics.loc["f1"].values[0],
    metrics.loc["roc_auc"].values[0],
]
colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6"]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(metric_names, metric_vals, color=colors, edgecolor="black")
for bar, val in zip(bars, metric_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("XGBoost + SMOTE Performance Metrics (SNP only)", fontsize=14)
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(OUT_DIR / "figure_metrics.png", dpi=150)
plt.close()
print("  Figure 3: Metrics bar ✓")

# --- Figure 4: Comparison ---
metrics_list = ["accuracy", "precision", "recall", "f1", "roc_auc"]
labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
snp_vals = [snp_metrics.loc[m].values[0] for m in metrics_list]
combined_vals = [combined_metrics.loc[m].values[0] for m in metrics_list]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, snp_vals, width, label="SNP only", color="#3498db")
bars2 = ax.bar(x + width/2, combined_vals, width, label="SNP + k-mer", color="#e74c3c")

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f"{height:.3f}", ha="center", va="bottom", fontsize=10)

ax.set_ylabel("Score", fontsize=12)
ax.set_title("SNP only vs SNP + k-mer", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(OUT_DIR / "figure_comparison.png", dpi=150)
plt.close()
print("  Figure 4: Comparison ✓")

print(f"\n✓ All figures saved to {OUT_DIR}")