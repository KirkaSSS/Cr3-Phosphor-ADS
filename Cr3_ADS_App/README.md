# Cr³⁺ Phosphor Algorithmic Decision System (ADS)

Multi-model ADS for predictive screening of Cr³⁺-doped NIR-emitting phosphors.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Models
| Category | Description |
|---|---|
| 🌳 Gradient Boosting | CatBoost / XGBoost / LightGBM |
| 📊 Gaussian Process | ARD kernel, calibrated uncertainty |
| 🧠 ANN (MLP) | Optuna-optimized architecture |

## Upload your trained model
Place your `.cbm`, `.pkl` or `.joblib` model file via the sidebar uploader.
Without a model file the app runs in **demo mode** with simulated predictions.

## Authors
Snežana Đurković · ORCID: 0009-0007-6638-0682
OMAS Group, Institute of Nuclear Sciences "Vinča", University of Belgrade
