# Cr³⁺ Phosphor Algorithmic Decision System (ADS)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cr3-phosphor-ads.streamlit.app)

**Multi-model ADS for predictive screening of Cr³⁺-doped NIR-emitting phosphors.**

Predicts the crystal field parameter **Dq/B** and ranks candidates into **Tier 1–4** for rational synthesis prioritization.

---

## Model categories

| Category | Description | CV R² | CV MAE |
|---|---|---|---|
| 🌳 Gradient Boosting | CatBoost / XGBoost / LightGBM | 0.60 | 0.164 |
| 📊 Gaussian Process | ARD kernel, calibrated uncertainty | 0.53 | 0.171 |
| 🧠 ANN (MLP) | Optuna-optimized architecture | 0.51 | 0.170 |

---

## Tier-based decision stratification

| Tier | Condition | Decision output |
|---|---|---|
| Tier 1 — Strong | Dq/B ∈ [2.2, 2.8], σ < 0.20 | Highest priority for synthesis |
| Tier 2 — Promising | Dq/B ∈ [2.2, 2.8], σ < 0.40 | Recommended for synthesis |
| Tier 3 — Uncertain/Edge | In range but uncertain, or near boundary | Requires theoretical analysis |
| Tier 4 — Out of range | Outside target NIR window | Excluded from synthesis pipeline |

---

## Dataset

- **243** experimentally characterized Cr³⁺-doped phosphors
- **207** training / **36** hold-out validation (one per major oxide SGR family)
- **15** structural and chemical descriptors
- Dq/B range: 1.17 – 3.43

---

## Usage

Upload a `.xlsx` or `.csv` file with candidate compositions, or enter a single compound manually. The ADS returns Tier-ranked predictions with uncertainty estimates.

Without a trained model file the app runs in **demo mode** with simulated predictions. Upload your `.cbm`, `.pkl` or `.joblib` model via the sidebar.

---

## Repositories

| Model | Repository |
|---|---|
| Gradient Boosting (CatBoost) | [Cr3-CatBoost-Expert-System](https://github.com/KirkaSSS/Cr3-CatBoost-Expert-System) |
| GBM Comparison | [Cr3-GBM-Comparison](https://github.com/KirkaSSS/Cr3-GBM-Comparison) |
| Gaussian Process | [Cr3-GP-Expert-System](https://github.com/KirkaSSS/Cr3-GP-Expert-System) |
| ANN (MLP) | [Cr3-ANN-Expert-System](https://github.com/KirkaSSS/Cr3-ANN-Expert-System) |
| Full pipeline | [Photoluminescent-Material-AI-Expert-System](https://github.com/KirkaSSS/Photoluminescent-Material-AI-Expert-System) |

---

## References

- Chiusi, F. et al. (2020). *Automating Society Report 2020.* AlgorithmWatch & Bertelsmann Stiftung.
- Koeszegi, S.T. (ed.) (2024). *AI @ Work: Human Empowerment or Disempowerment?* Springer.
- Gebru, T. et al. (2018). *Datasheets for Datasets.* arXiv:1803.09010.

---

## Authors

**Snežana Đurković** · ORCID: [0009-0007-6638-0682](https://orcid.org/0009-0007-6638-0682)  
Prof. Dr. Miroslav Dramićanin · Dr. Zoran Ristić  
OMAS Group, Institute of Nuclear Sciences "Vinča", University of Belgrade

---

MIT License · Belgrade, 2026
