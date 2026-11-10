"""
ساخت شکل‌های مقاله
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix

OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# بارگذاری
metrics = pd.read_csv("results/xgboost_azzato/metrics.tsv", sep="\t", index_col=0)
preds = pd.read_csv("results/xgboost_azzato/predictions.tsv", sep="\t")
gwas = pd.read_csv("results/gwas_azzato/gwas_results.tsv", sep="\t")

# --- Figure 1: ROC Curve ---
print("Figure 1: ROC curve...")
y_true = preds["y_true"].values
y_proba = preds["y_proba"].values
fpr, tpr, _ = roc_curve(y_true, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random (AUC = 0.5)")
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

# --- Figure 2: Confusion Matrix ---
print("Figure 2: Confusion matrix...")
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
                color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=16, fontweight="bold")
plt.colorbar(im)
plt.tight_layout()
plt.savefig(OUT_DIR / "figure_confusion_matrix.png", dpi=150)
plt.close()

# --- Figure 3: GWAS Manhattan Plot ---
print("Figure 3: GWAS Manhattan plot...")
gwas["log_p"] = -np.log10(gwas["p_value"].clip(lower=1e-50))
gwas_sorted = gwas.sort_values("pos")

fig, ax = plt.subplots(figsize=(14, 6))
ax.scatter(gwas_sorted["pos"], gwas_sorted["log_p"], alpha=0.5, s=10, c="steelblue")
ax.axhline(-np.log10(0.05), color="orange", linestyle="--", label="P<0.05")
ax.axhline(-np.log10(0.001), color="red", linestyle="--", label="P<0.001")
ax.set_xlabel("Genomic position (bp)", fontsize=12)
ax.set_ylabel("-log10(P)", fontsize=12)
ax.set_title("GWAS Manhattan Plot — Azzato 2026", fontsize=14)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "figure_manhattan.png", dpi=150)
plt.close()

# --- Figure 4: Performance Metrics Bar Chart ---
print("Figure 4: Performance metrics...")
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
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
ax.set_ylabel("Score", fontsize=12)
ax.set_title("XGBoost + SMOTE Performance Metrics", fontsize=14)
ax.legend()
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(OUT_DIR / "figure_metrics.png", dpi=150)
plt.close()

# --- Figure 5: Probability distribution ---
print("Figure 5: Probability distribution...")
fig, ax = plt.subplots(figsize=(10, 6))
for label, color in [(0, "#3498db"), (1, "#e74c3c")]:
    subset = preds[preds["y_true"] == label]["y_proba"]
    ax.hist(subset, bins=30, alpha=0.6, color=color,
            label=f"{'Susceptible' if label == 0 else 'Resistant'} (n={len(subset)})")
ax.axvline(0.5, color="black", linestyle="--", label="Threshold = 0.5")
ax.set_xlabel("Predicted probability", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.set_title("Predicted Probability Distribution", fontsize=14)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "figure_probability.png", dpi=150)
plt.close()

print(f"\n✓ All figures saved to {OUT_DIR}")