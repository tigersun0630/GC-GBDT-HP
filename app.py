#!/usr/bin/env python
# coding: utf-8

import os
import sys
import io
import base64
import joblib
import shap
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as components


# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="H. pylori Genetic Variation-Driven Gastric Cancer Risk Prediction",
    layout="wide"
)


# =========================
# Custom CSS
# =========================
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1280px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    .main-title {
        font-size: 2.25rem;
        font-weight: 800;
        line-height: 1.25;
        color: #2f3340;
        margin-bottom: 0.4rem;
    }

    .subtitle {
        font-size: 1.0rem;
        color: #666b7a;
        margin-bottom: 1.5rem;
    }

    div[data-testid="stForm"] {
        border: 1px solid #d9dce3;
        border-radius: 14px;
        padding: 24px 26px 20px 26px;
        background-color: #ffffff;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }

    div[data-testid="stSelectbox"] label {
        font-size: 0.92rem;
        font-weight: 650;
        color: #252936;
    }

    div[data-testid="stSelectbox"] {
        margin-bottom: 0.35rem;
    }

    div[data-testid="stFormSubmitButton"] button {
        border-radius: 8px;
        height: 2.8rem;
        font-weight: 650;
    }

    .result-card {
        border: 1px solid #d9dce3;
        border-radius: 14px;
        padding: 22px 26px;
        margin-top: 1.4rem;
        background-color: #ffffff;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }

    .result-title {
        font-size: 1.25rem;
        font-weight: 750;
        color: #2f3340;
        margin-bottom: 0.6rem;
    }

    .result-line {
        font-size: 1.15rem;
        font-weight: 650;
        color: #222633;
        margin: 0.35rem 0;
    }

    .shap-card {
        border: 1px solid #d9dce3;
        border-radius: 14px;
        padding: 20px 22px;
        margin-top: 1.2rem;
        background-color: #ffffff;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 750;
        color: #2f3340;
        margin-bottom: 1rem;
    }

    .small-note {
        font-size: 0.9rem;
        color: #6b7280;
        margin-top: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="main-title">
        H. pylori Genetic Variation-Driven Gastric Cancer Risk Prediction:
        A SHAP-Explained Online Platform
    </div>
    <div class="subtitle">
        Select the absence or presence status of each genetic variation, then click Predict to obtain model-based risk estimation and SHAP explanation.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================
# absence / presence mapping
# =========================
OPTION_LIST = ["absence", "presence"]

OPTION_MAP = {
    "absence": 0,
    "presence": 1
}


# =========================
# Load model
# =========================
@st.cache_resource
def load_model():
    """
    Load GBDT.pkl.
    If model deserialization raises No module named '_loss',
    automatically apply sklearn._loss._loss compatibility mapping.
    """
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GBDT.pkl")

    try:
        model = joblib.load(model_path)
        return model

    except ModuleNotFoundError as e:
        if str(e) == "No module named '_loss'" or getattr(e, "name", None) == "_loss":
            import sklearn._loss._loss as cy_loss
            sys.modules["_loss"] = cy_loss
            model = joblib.load(model_path)
            return model
        else:
            raise e


@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)


try:
    clf = load_model()
except Exception as e:
    st.error(
        "Model loading failed. Please check whether GBDT.pkl is in the same directory as app.py "
        "and whether the scikit-learn version is compatible."
    )
    st.exception(e)
    st.stop()


# =========================
# Get and clean feature names
# =========================
if not hasattr(clf, "feature_names_in_"):
    st.error(
        "The current model does not have the feature_names_in_ attribute, "
        "so input fields cannot be generated automatically."
    )
    st.stop()

# Model original feature names may contain hidden spaces.
model_feature_names = [str(x) for x in clf.feature_names_in_]
clean_feature_names = [x.strip() for x in model_feature_names]

if len(clean_feature_names) != len(set(clean_feature_names)):
    duplicated = pd.Series(clean_feature_names)[pd.Series(clean_feature_names).duplicated()].unique().tolist()
    st.error(
        "After stripping hidden spaces, duplicated feature names were detected: "
        + ", ".join(duplicated)
    )
    st.stop()

# Clean name -> original model name
feature_to_model_name = dict(zip(clean_feature_names, model_feature_names))


# =========================
# SHAP-based display order
# =========================
display_order = [
    "omp32 T337H",
    "HP_1216 N549D",
    "cag13 S93N",
    "group_7765",
    "envA I134V",
    "HP_1117 A216S",
    "vacA_1",
    "HP_0595 A133T",
    "HP_1079 R328K",
    "HP_0610 K1244Q",
    "dhs1 Ter450",
    "HP_0276 R180K",
    "HP_1216 M444V",
    "group_13606",
    "rocC",
    "HP_0655 K520R",
    "group_5551",
    "pyrC_2",
    "group_240",
    "msrAB",
    "group_827",
    "infB G607D",
    "cagT"
]


# =========================
# Validate display order
# =========================
missing_in_model = [x for x in display_order if x not in feature_to_model_name]
extra_in_model = [x for x in clean_feature_names if x not in display_order]

if missing_in_model:
    st.error(
        "The following features in display_order were not found in the model: "
        + ", ".join(missing_in_model)
    )
    st.stop()

if extra_in_model:
    st.error(
        "The model contains extra features not included in display_order: "
        + ", ".join(extra_in_model)
    )
    st.stop()

if len(display_order) != 23:
    st.warning(
        f"The display_order currently contains {len(display_order)} features, not 23. "
        "Please confirm whether this is expected."
    )


# =========================
# Initialize input state
# =========================
def init_input_state():
    for name in display_order:
        key = f"input_{name}"
        if key not in st.session_state:
            st.session_state[key] = "absence"
        if st.session_state[key] not in OPTION_LIST:
            st.session_state[key] = "absence"


def reset_inputs():
    for name in display_order:
        st.session_state[f"input_{name}"] = "absence"


init_input_state()


# =========================
# Input form: two-column layout
# =========================
with st.form("genetic_variation_form", clear_on_submit=False):

    st.markdown('<div class="section-title">Genetic Variation Input</div>', unsafe_allow_html=True)

    # Two-column row-wise layout:
    # row 1: feature 1 + feature 2
    # row 2: feature 3 + feature 4
    # ...
    for i in range(0, len(display_order), 2):
        col1, col2 = st.columns(2, gap="large")

        with col1:
            name = display_order[i]
            st.selectbox(
                label=name,
                options=OPTION_LIST,
                key=f"input_{name}"
            )

        with col2:
            if i + 1 < len(display_order):
                name = display_order[i + 1]
                st.selectbox(
                    label=name,
                    options=OPTION_LIST,
                    key=f"input_{name}"
                )
            else:
                st.empty()

    st.markdown("---")

    btn_col1, btn_col2, btn_col3 = st.columns([1.2, 1.2, 6])

    with btn_col1:
        predict_btn = st.form_submit_button(
            "Predict",
            type="primary",
            use_container_width=True
        )

    with btn_col2:
        reset_btn = st.form_submit_button(
            "Reset",
            use_container_width=True,
            on_click=reset_inputs
        )


# =========================
# Prediction and SHAP explanation
# =========================
if predict_btn:

    # Original dropdown selections: absence / presence
    input_label_dict = {
        name: st.session_state[f"input_{name}"]
        for name in display_order
    }

    # Numeric values for display order
    input_numeric_display = {
        name: OPTION_MAP[st.session_state[f"input_{name}"]]
        for name in display_order
    }

    # Numeric values using original model feature names
    input_numeric_model = {
        feature_to_model_name[name]: input_numeric_display[name]
        for name in display_order
    }

    # X_model must use the exact feature names and order stored in the model
    X_model = pd.DataFrame(
        [[input_numeric_model[col] for col in model_feature_names]],
        columns=model_feature_names
    )

    # X_display is only for showing clean feature names to users
    X_label_display = pd.DataFrame(
        [[input_label_dict[name] for name in display_order]],
        columns=display_order
    )

    X_numeric_display = pd.DataFrame(
        [[input_numeric_display[name] for name in display_order]],
        columns=display_order
    )

    # =========================
    # Model prediction
    # =========================
    try:
        pred_class = clf.predict(X_model)[0]

        pred_proba = None
        target_class = None
        target_index = 0

        if hasattr(clf, "predict_proba"):
            pred_proba_all = clf.predict_proba(X_model)
            classes = list(clf.classes_)

            # Default target: class 1.
            # If class 1 does not exist, use the last class.
            if 1 in classes:
                target_class = 1
                target_index = classes.index(1)
            else:
                target_class = classes[-1]
                target_index = len(classes) - 1

            pred_proba = float(pred_proba_all[0][target_index])

        else:
            pred_proba_all = None
            classes = None

    except Exception as e:
        st.error("Model prediction failed.")
        st.exception(e)
        st.stop()


    # =========================
    # Result card
    # =========================
    st.markdown(
        '<div class="result-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="result-title">Prediction Results</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="result-line">Predicted class: {pred_class}</div>',
        unsafe_allow_html=True
    )

    if pred_proba is not None:
        st.markdown(
            f'<div class="result-line">Prediction probability for class {target_class}: {pred_proba * 100:.2f}%</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="small-note">The current model does not support predict_proba, so prediction probability cannot be displayed.</div>',
            unsafe_allow_html=True
        )

    with st.expander("View current input data", expanded=False):
        st.markdown("Original input: absence / presence")
        st.dataframe(X_label_display, use_container_width=True)

        st.markdown("Model input values: absence = 0, presence = 1")
        st.dataframe(X_numeric_display, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


    # =========================
    # SHAP force plot
    # =========================
    st.markdown(
        '<div class="shap-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">SHAP Force Plot</div>',
        unsafe_allow_html=True
    )

    try:
        explainer = load_explainer(clf)
        shap_values = explainer(X_model)

        values = shap_values.values
        base_values = shap_values.base_values

        # Compatible with binary classification, multiclass classification, and different SHAP versions
        if values.ndim == 3:
            # shape: [number of samples, number of features, number of classes]
            shap_value_single = values[0, :, target_index]

            if np.ndim(base_values) == 2:
                expected_value = base_values[0, target_index]
            elif np.ndim(base_values) == 1:
                expected_value = base_values[target_index]
            else:
                expected_value = base_values

        elif values.ndim == 2:
            # shape: [number of samples, number of features]
            shap_value_single = values[0]

            if np.ndim(base_values) == 1:
                expected_value = base_values[0]
            else:
                expected_value = base_values

        else:
            st.error(f"Unrecognized SHAP values dimension: {values.shape}")
            st.stop()

        expected_value = float(np.ravel(expected_value)[0])

        # Use clean feature names in model original order for the SHAP plot
        clean_model_order_names = [x.strip() for x in model_feature_names]
        feature_values_for_plot = np.round(
            X_model.iloc[0].values.astype(float),
            3
        )

        plt.close("all")

        shap.force_plot(
            expected_value,
            np.round(shap_value_single, 3),
            feature_values_for_plot,
            feature_names=clean_model_order_names,
            figsize=(42, 4.2),
            matplotlib=True,
            show=False,
            text_rotation=0,
            contribution_threshold=0.03
        )

        fig = plt.gcf()

        svg_buffer = io.StringIO()
        fig.savefig(
            svg_buffer,
            format="svg",
            bbox_inches="tight"
        )
        plt.close(fig)

        svg_data = svg_buffer.getvalue()
        b64 = base64.b64encode(svg_data.encode("utf-8")).decode("utf-8")

        components.html(
            f"""
            <div style="
                width:100%;
                overflow-x:auto;
                border:1px solid #eef0f4;
                border-radius:12px;
                padding:12px;
                background:#ffffff;
            ">
                <img src="data:image/svg+xml;base64,{b64}"
                     style="width:100%; min-width:1200px;">
            </div>
            """,
            height=380,
            scrolling=False
        )

    except Exception as e:
        st.error("Failed to generate SHAP force plot.")
        st.exception(e)

    st.markdown('</div>', unsafe_allow_html=True)


else:
    st.info("After selecting the genetic variation status, click Predict. The prediction result and SHAP explanation will appear below the input panel.")
