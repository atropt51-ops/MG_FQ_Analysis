# Analysis Code for FQ Resistance in *Mycoplasma genitalium*

This repository contains the analysis pipeline and code for the manuscript:
> *Machine learning recapitulates genotype-defined fluoroquinolone resistance in Mycoplasma genitalium and prioritizes non-canonical genomic candidates*

## Overview
This project combines GWAS, XGBoost machine learning, SHAP interpretation, and molecular docking to investigate fluoroquinolone (FQ) resistance in *M. genitalium*. Phenotype was genotype-defined based on parC/gyrA mutations (BioProject PRJNA1367946).

## Requirements
- Python 3.10+
- Packages: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `shap`, `matplotlib`, `biopython`, `scipy`
- External tools (for docking): `AutoDock Vina` v1.2.7, `Open Babel` v3.1.1

## Key Scripts
- `27_xgboost_without_known_markers.py`: Sensitivity analysis (removing parC S83I and QRDR)
- `28_ld_clumping.py`: LD-based clumping to identify independent loci
- `29_tree_based_validation.py`: Lineage-aware GroupKFold validation (BAPS groups)
- `30_gwas_af_ci_gc.py`: GWAS with allele frequency, 95% CI, and genomic control
- `31_imbalance_comparison.py`: Comparison of SMOTE, class weighting, and baseline
- `32_kmer_only_model.py`: k-mer vs SNP feature comparison
- `33_noncanonical_variant_table.py`: Annotation of prioritized non-canonical variants
- `34_shap_direction.py`: Signed SHAP effects (risk vs protective)
- `35_download_structures.py`: Download ParC/GyrA structures from AlphaFold DB
- `36_apply_S83I_mutation.py`: Apply S83I mutation to ParC structure
- `37_docking.sh`: (If applicable) Commands for AutoDock Vina docking

## Reproducibility
All scripts are numbered to reflect the pipeline order. Results are saved in the `results/` directory. The full dataset (SNP matrix, phenotype table) is derived from BioProject PRJNA1367946.
