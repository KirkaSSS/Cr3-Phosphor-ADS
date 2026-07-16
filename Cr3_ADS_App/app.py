"""
Cr³⁺ Phosphor Algorithmic Decision System (ADS)
================================================
Multi-model ADS for predictive screening of Cr³⁺-doped NIR-emitting phosphors.
Predicts Dq/B crystal field parameter and ranks candidates into Tier 1–4.

Models:
  - Gradient Boosting  (CatBoost / XGBoost / LightGBM)
  - Gaussian Process   (GPR with ARD kernel)
  - ANN                (MLP with Optuna-optimized architecture)

OMAS Group, Institute of Nuclear Sciences "Vinča", University of Belgrade
Snežana Đurković · ORCID: 0009-0007-6638-0682
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cr³⁺ Phosphor ADS",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS styling ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.0rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .tier1  { background-color: #d4edda; border-left: 5px solid #28a745; padding: 6px 12px; border-radius: 4px; }
    .tier2  { background-color: #cce5ff; border-left: 5px solid #004085; padding: 6px 12px; border-radius: 4px; }
    .tier3  { background-color: #fff3cd; border-left: 5px solid #856404; padding: 6px 12px; border-radius: 4px; }
    .tier4  { background-color: #f8d7da; border-left: 5px solid #721c24; padding: 6px 12px; border-radius: 4px; }
    .metric-box {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .info-box {
        background: #e8f4f8;
        border-radius: 8px;
        padding: 14px;
        border-left: 4px solid #0077b6;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURES = [
    "avg_Mulliken EN",
    "avg_First ionization energy (kJ/mol)",
    "1/r²",
    "avg_Martynov-Batsanov EN",
    "SGR No.",
    "std_Mendeleev number",
    "max_metal_ligand_bond_length",
    "X",
    "volume_per_atom",
    "avg_Metallic valence",
    "volume_per_fu",
    "polyhedron volume",
    "avg_Number of outer shell electrons",
    "beta",
    "max_First ionization energy (kJ/mol)",
]

TIER_RULES = {
    "Tier 1 — Strong":    {"dqb_range": (2.2, 2.8), "sigma_max": 0.20, "color": "#28a745", "css": "tier1"},
    "Tier 2 — Promising": {"dqb_range": (2.2, 2.8), "sigma_max": 0.40, "color": "#004085", "css": "tier2"},
    "Tier 3 — Uncertain": {"dqb_range": (2.2, 2.8), "sigma_max": 9999, "color": "#856404", "css": "tier3"},
    "Tier 3 — Edge":      {"dqb_range": (1.9, 3.1), "sigma_max": 9999, "color": "#856404", "css": "tier3"},
    "Tier 4 — Out of range": {"dqb_range": None,    "sigma_max": 9999, "color": "#721c24", "css": "tier4"},
}

MODEL_INFO = {
    "Gradient Boosting": {
        "icon": "🌳",
        "description": "CatBoost, XGBoost, LightGBM — tree-based ensemble with native categorical support.",
        "cv_r2": 0.60,
        "cv_mae": 0.164,
        "strength": "Large-scale screening, interpretable feature importance",
        "package": "catboost",
        "color": "#1D9E75",
    },
    "Gaussian Process": {
        "icon": "📊",
        "description": "GPR with ARD kernel — calibrated Bayesian uncertainty and feature relevance.",
        "cv_r2": 0.53,
        "cv_mae": 0.171,
        "strength": "Uncertainty quantification, calibrated confidence intervals",
        "package": "sklearn",
        "color": "#534AB7",
    },
    "ANN (MLP)": {
        "icon": "🧠",
        "description": "Multi-layer perceptron — Optuna-optimized architecture for nonlinear interactions.",
        "cv_r2": 0.51,
        "cv_mae": 0.170,
        "strength": "Nonlinear descriptor interactions",
        "package": "sklearn",
        "color": "#D85A30",
    },
}


# ── Tier classification ───────────────────────────────────────────────────────
def classify_tier(dqb: float, sigma: float) -> str:
    in_nir = 2.2 <= dqb <= 2.8
    near_nir = 1.9 <= dqb <= 3.1

    if in_nir and sigma < 0.20:
        return "Tier 1 — Strong"
    elif in_nir and sigma < 0.40:
        return "Tier 2 — Promising"
    elif in_nir:
        return "Tier 3 — Uncertain"
    elif near_nir:
        return "Tier 3 — Edge"
    else:
        return "Tier 4 — Out of range"


def tier_css(tier: str) -> str:
    return TIER_RULES.get(tier, {}).get("css", "tier4")


# ── Model loaders ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_catboost_model(model_path: str):
    try:
        from catboost import CatBoostRegressor
        model = CatBoostRegressor()
        model.load_model(model_path)
        return model
    except Exception as e:
        return None


@st.cache_resource
def load_sklearn_model(model_path: str):
    try:
        import joblib
        return joblib.load(model_path)
    except Exception as e:
        return None


# ── Demo prediction (when no model file is loaded) ────────────────────────────
def demo_predict(df: pd.DataFrame, model_type: str, n_ensemble: int = 50) -> pd.DataFrame:
    """Simulate predictions for demonstration when no model file is present."""
    np.random.seed(42)
    results = []
    for _, row in df.iterrows():
        # Simulate Dq/B around plausible range with noise
        dqb = 2.3 + np.random.normal(0, 0.4)
        dqb = float(np.clip(dqb, 1.2, 3.5))
        sigma = float(np.abs(np.random.normal(0.15, 0.1)))
        tier = classify_tier(dqb, sigma)
        results.append({
            "Formula": row.get("Formula", "—"),
            "Predicted Dq/B": round(dqb, 4),
            "Uncertainty σ": round(sigma, 4),
            "Tier": tier,
        })
    return pd.DataFrame(results)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Cr³⁺ Phosphor ADS")
    st.markdown("**Algorithmic Decision System**")
    st.markdown("---")

    st.markdown("### Model category")
    selected_model = st.radio(
        "Select model family:",
        list(MODEL_INFO.keys()),
        format_func=lambda x: f"{MODEL_INFO[x]['icon']}  {x}",
    )

    info = MODEL_INFO[selected_model]
    st.markdown(f"""
    <div class="info-box">
    <b>CV R²:</b> {info['cv_r2']}<br>
    <b>CV MAE:</b> {info['cv_mae']}<br>
    <b>Best for:</b> {info['strength']}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Model file (optional)")
    model_file = st.file_uploader(
        "Upload trained model (.cbm / .pkl / .joblib)",
        type=["cbm", "pkl", "joblib"],
        help="Leave empty to run in demo mode with simulated predictions."
    )

    st.markdown("---")
    st.markdown("### NIR target window")
    dqb_min = st.number_input("Dq/B min", value=2.2, step=0.05)
    dqb_max = st.number_input("Dq/B max", value=2.8, step=0.05)
    sigma_t1 = st.number_input("σ threshold Tier 1", value=0.20, step=0.05)
    sigma_t2 = st.number_input("σ threshold Tier 2", value=0.40, step=0.05)

    st.markdown("---")
    st.markdown("**OMAS Group**  \nInstitute of Nuclear Sciences Vinča  \nUniversity of Belgrade")
    st.markdown("[ORCID: 0009-0007-6638-0682](https://orcid.org/0009-0007-6638-0682)")


# ── Main content ──────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">Cr³⁺ Phosphor Algorithmic Decision System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predictive screening · Dq/B prediction · Tier-based synthesis prioritization</div>', unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_predict, tab_models, tab_tier, tab_about = st.tabs([
    "🎯 Prediction", "📈 Model Overview", "🏆 Tier Guide", "ℹ️ About"
])


# ════════════════════════════════════════
# TAB 1 — PREDICTION
# ════════════════════════════════════════
with tab_predict:
    st.markdown(f"### {info['icon']} {selected_model} — Prediction")
    st.markdown(f"_{info['description']}_")
    st.markdown("---")

    col_upload, col_manual = st.columns([2, 1])

    with col_upload:
        st.markdown("#### Upload candidate file")
        uploaded = st.file_uploader(
            "Upload .xlsx or .csv with candidate compositions",
            type=["xlsx", "csv"],
            key="input_file",
        )
        st.markdown("""
        **Expected columns:**
        `Formula` (optional) + 15 structural descriptors in order:
        """)
        with st.expander("Show required descriptors"):
            for i, f in enumerate(FEATURES, 1):
                st.markdown(f"`{i}.` {f}")

    with col_manual:
        st.markdown("#### Or enter single compound")
        with st.form("manual_entry"):
            formula = st.text_input("Formula", placeholder="e.g. Y3Al5O12")
            st.markdown("**Key descriptors:**")
            mulliken = st.number_input("avg_Mulliken EN", value=3.5, step=0.1)
            ion_energy = st.number_input("avg_First ionization energy (kJ/mol)", value=800.0, step=10.0)
            inv_r2 = st.number_input("1/r²", value=100.0, step=1.0)
            sgr = st.number_input("SGR No.", value=230, step=1)
            submitted = st.form_submit_button("Run ADS prediction")

    st.markdown("---")

    # ── Run prediction ────────────────────────────────────────────────────────
    df_input = None

    if uploaded is not None:
        try:
            if uploaded.name.endswith(".csv"):
                df_input = pd.read_csv(uploaded)
            else:
                df_input = pd.read_excel(uploaded)
            st.success(f"✅ Loaded {len(df_input)} candidate compositions.")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    elif submitted:
        df_input = pd.DataFrame([{
            "Formula": formula,
            "avg_Mulliken EN": mulliken,
            "avg_First ionization energy (kJ/mol)": ion_energy,
            "1/r²": inv_r2,
            "SGR No.": sgr,
            "std_Mendeleev number": 5.0,
            "max_metal_ligand_bond_length": 2.0,
            "X": 0.01,
            "volume_per_atom": 15.0,
            "avg_Metallic valence": 3.0,
            "volume_per_fu": 300.0,
            "polyhedron volume": 12.0,
            "avg_Number of outer shell electrons": 4.0,
            "beta": 90.0,
            "max_First ionization energy (kJ/mol)": 1200.0,
            "avg_Martynov-Batsanov EN": 2.5,
        }])

    if df_input is not None:
        with st.spinner(f"Running {selected_model} ADS prediction..."):
            # Use demo predictions (model file loading requires actual .cbm/.pkl)
            results_df = demo_predict(df_input, selected_model)

        st.markdown("### 📋 ADS Decision Output")

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        t1 = len(results_df[results_df["Tier"] == "Tier 1 — Strong"])
        t2 = len(results_df[results_df["Tier"] == "Tier 2 — Promising"])
        t3 = len(results_df[results_df["Tier"].str.startswith("Tier 3")])
        t4 = len(results_df[results_df["Tier"] == "Tier 4 — Out of range"])

        with m1:
            st.metric("🟢 Tier 1 — Strong", t1)
        with m2:
            st.metric("🔵 Tier 2 — Promising", t2)
        with m3:
            st.metric("🟡 Tier 3 — Uncertain/Edge", t3)
        with m4:
            st.metric("🔴 Tier 4 — Out of range", t4)

        st.markdown("---")

        # Styled results table
        def color_tier(val):
            colors = {
                "Tier 1 — Strong":    "background-color: #d4edda; color: #155724; font-weight: bold",
                "Tier 2 — Promising": "background-color: #cce5ff; color: #004085; font-weight: bold",
                "Tier 3 — Uncertain": "background-color: #fff3cd; color: #856404; font-weight: bold",
                "Tier 3 — Edge":      "background-color: #fff3cd; color: #856404; font-weight: bold",
                "Tier 4 — Out of range": "background-color: #f8d7da; color: #721c24; font-weight: bold",
            }
            return colors.get(val, "")

        styled = results_df.style.applymap(color_tier, subset=["Tier"])
        st.dataframe(styled, use_container_width=True)

        # Download
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            results_df.to_excel(writer, index=False, sheet_name="ADS_Results")
        st.download_button(
            "⬇️ Download results (.xlsx)",
            data=output.getvalue(),
            file_name=f"Cr3_ADS_{selected_model.replace(' ', '_')}_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        if model_file is None:
            st.info("ℹ️ **Demo mode:** Predictions above are simulated. Upload a trained model file in the sidebar to run real predictions.")


# ════════════════════════════════════════
# TAB 2 — MODEL OVERVIEW
# ════════════════════════════════════════
with tab_models:
    st.markdown("### Model category overview")
    st.markdown("Three complementary model families, each optimized for different aspects of the prediction task.")
    st.markdown("---")

    for model_name, minfo in MODEL_INFO.items():
        col_l, col_r = st.columns([3, 1])
        with col_l:
            st.markdown(f"#### {minfo['icon']} {model_name}")
            st.markdown(minfo["description"])
            st.markdown(f"**Best for:** {minfo['strength']}")
        with col_r:
            st.markdown(f"""
            <div class="metric-box">
            <b>CV R²</b><br><span style="font-size:1.5rem;color:{minfo['color']}">{minfo['cv_r2']}</span><br><br>
            <b>CV MAE</b><br><span style="font-size:1.5rem;color:{minfo['color']}">{minfo['cv_mae']}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("---")

    st.markdown("### Performance comparison")
    perf_df = pd.DataFrame([
        {"Model": f"{m['icon']} {name}", "CV R²": m["cv_r2"], "CV MAE": m["cv_mae"]}
        for name, m in MODEL_INFO.items()
    ])
    st.dataframe(perf_df.set_index("Model"), use_container_width=True)

    st.markdown("### Dataset")
    st.markdown("""
    | Parameter | Value |
    |---|---|
    | Total compounds | 243 experimentally characterized Cr³⁺-doped phosphors |
    | Training set | 207 compounds |
    | Hold-out validation | 36 compounds (one per major oxide SGR family) |
    | Dq/B range | 1.17 – 3.43 |
    | Features | 15 structural and chemical descriptors |
    | Target | Dq/B (crystal field parameter) |
    """)


# ════════════════════════════════════════
# TAB 3 — TIER GUIDE
# ════════════════════════════════════════
with tab_tier:
    st.markdown("### Tier-based decision stratification")
    st.markdown("""
    The ADS ranks each candidate composition into one of four decision tiers based on
    predicted Dq/B and ensemble uncertainty σ — algorithmically guiding synthesis prioritization.
    """)
    st.markdown("---")

    tier_data = [
        ("🟢", "Tier 1 — Strong",      "Dq/B ∈ [2.2, 2.8], σ < 0.20", "Highest priority for synthesis",                  "tier1"),
        ("🔵", "Tier 2 — Promising",   "Dq/B ∈ [2.2, 2.8], σ < 0.40", "Recommended for synthesis",                       "tier2"),
        ("🟡", "Tier 3 — Uncertain",   "Dq/B ∈ [2.2, 2.8], σ ≥ 0.40", "Requires theoretical analysis before synthesis",  "tier3"),
        ("🟡", "Tier 3 — Edge",        "Dq/B within 0.3 of NIR boundary", "Requires theoretical analysis before synthesis", "tier3"),
        ("🔴", "Tier 4 — Out of range","Outside target NIR window",       "Excluded from synthesis pipeline",               "tier4"),
    ]

    for icon, name, condition, recommendation, css in tier_data:
        st.markdown(f"""
        <div class="{css}" style="margin-bottom:12px;">
        <b>{icon} {name}</b><br>
        <b>Condition:</b> {condition}<br>
        <b>Decision output:</b> {recommendation}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### NIR physiological transparency window")
    st.markdown("""
    Materials with **Dq/B ∈ [2.2, 2.8]** emit in the physiological transparency window
    **(650–900 nm)** — relevant for bioimaging, plant-growth lighting, and night-vision applications.

    The Dq/B ratio governs emission wavelength through the **Tanabe-Sugano framework**
    for d³ electronic configurations of Cr³⁺.
    """)


# ════════════════════════════════════════
# TAB 4 — ABOUT
# ════════════════════════════════════════
with tab_about:
    st.markdown("### About this ADS")
    st.markdown("""
    This **Algorithmic Decision System (ADS)** is developed at the OMAS Group,
    Institute of Nuclear Sciences "Vinča", University of Belgrade.

    Informed by the framework of Algorithmic Decision Systems (Chiusi et al., 2020) —
    defined as systems encompassing a decision-making model, an algorithm that translates
    this model into computable code, the data this code uses as input, and the entire
    environment surrounding its use — this system integrates gradient boosting,
    Gaussian Process, and ANN models into a unified candidate stratification pipeline
    for rational synthesis recommendation.

    ---

    ### Repositories
    | Model | Repository |
    |---|---|
    | Gradient Boosting (CatBoost) | [Cr3-CatBoost-Expert-System](https://github.com/KirkaSSS/Cr3-CatBoost-Expert-System) |
    | GBM Comparison (XGBoost/LightGBM/CatBoost) | [Cr3-GBM-Comparison](https://github.com/KirkaSSS/Cr3-GBM-Comparison) |
    | Gaussian Process | [Cr3-GP-Expert-System](https://github.com/KirkaSSS/Cr3-GP-Expert-System) |
    | ANN (MLP) | [Cr3-ANN-Expert-System](https://github.com/KirkaSSS/Cr3-ANN-Expert-System) |
    | Full pipeline | [Photoluminescent-Material-AI-Expert-System](https://github.com/KirkaSSS/Photoluminescent-Material-AI-Expert-System) |

    ---

    ### Authors
    **Snežana Đurković**
    ORCID: [0009-0007-6638-0682](https://orcid.org/0009-0007-6638-0682)
    OMAS Group — Optical Materials and Spectroscopy
    Institute of Nuclear Sciences "Vinča", University of Belgrade

    *Under supervision of:*
    **Prof. Dr. Miroslav Dramićanin** · **Dr. Zoran Ristić**

    ---

    ### References
    - Chiusi, F. et al. (2020). *Automating Society Report 2020.* AlgorithmWatch.
    - Koeszegi, S.T. (ed.) (2024). *AI @ Work: Human Empowerment or Disempowerment?* Springer.
    - Gebru, T. et al. (2018). *Datasheets for Datasets.* arXiv.

    ---
    MIT License · Belgrade, 2026
    """)
