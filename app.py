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
    page_title="H. pylori Genetic Variation-Driven Gastric Cancer Risk Prediction: A SHAP-Explained Online Platform",
    layout="wide"
)

st.title("H. pylori Genetic Variation-Driven Gastric Cancer Risk Prediction: A SHAP-Explained Online Platform")


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
    Load GBDT.pkl normally.
    If model deserialization raises No module named '_loss',
    automatically apply the sklearn._loss._loss compatibility mapping.
    """
    try:
        model = joblib.load("GBDT.pkl")
        return model

    except ModuleNotFoundError as e:
        if str(e) == "No module named '_loss'" or getattr(e, "name", None) == "_loss":
            import sklearn._loss._loss as cy_loss
            sys.modules["_loss"] = cy_loss
            model = joblib.load("GBDT.pkl")
            return model
        else:
            raise e


@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)


try:
    clf = load_model()
except Exception as e:
    st.error("Model loading failed. Please check whether GBDT.pkl is in the same directory as app.py and whether the scikit-learn version is compatible.")
    st.exception(e)
    st.stop()


# =========================
# Get feature names
# =========================
if not hasattr(clf, "feature_names_in_"):
    st.error("The current model does not have the feature_names_in_ attribute, so input fields cannot be generated automatically.")
    st.stop()

# The model may contain hidden leading/trailing spaces in feature names.
# Keep the original model feature names for prediction, and use stripped names for display.
feature_names_model = [str(x) for x in clf.feature_names_in_]
feature_names_display = [x.strip() for x in feature_names_model]

if len(feature_names_display) != len(set(feature_names_display)):
    st.error("After removing leading/trailing spaces, duplicated feature names were detected. Please check the model feature names.")
    st.stop()

# display name -> original model name; this preserves hidden spaces when calling clf.predict(X)
display_to_model = dict(zip(feature_names_display, feature_names_model))
# display name -> index used by Streamlit session_state keys
display_to_index = {name: i for i, name in enumerate(feature_names_display)}

if len(feature_names_display) != 23:
    st.warning(f"The current model has {len(feature_names_display)} features, not 23. Please confirm that the correct model is loaded.")


# =========================
# Feature display order
# Ordered according to the SHAP beeswarm plot, from high to low importance
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

# Validate after stripping hidden spaces
missing_features = set(feature_names_display) - set(display_order)
extra_features = set(display_order) - set(feature_names_display)

if missing_features:
    st.error(f"显示顺序列表中缺少以下特征：{', '.join(sorted(missing_features))}")
    with st.expander("Debug: model feature names"):
        st.write(feature_names_model)
    st.stop()

if extra_features:
    st.error(f"显示顺序列表中包含模型没有的特征：{', '.join(sorted(extra_features))}")
    with st.expander("Debug: model feature names"):
        st.write(feature_names_model)
    st.stop()


# =========================
# Initialize input state
# =========================
def init_input_state():
    """
    Initialize the dropdown state for each variable.
    The default value is absence for all variables.
    """
    for i, name in enumerate(feature_names_display):
        key = f"input_{i}"

        if key not in st.session_state:
            st.session_state[key] = "absence"

        # Prevent old number_input states such as 0.0 / 1.0 from causing selectbox errors
        if st.session_state[key] not in OPTION_LIST:
            st.session_state[key] = "absence"


def reset_inputs():
    """
    Reset all inputs to absence.
    """
    for i, name in enumerate(feature_names_display):
        st.session_state[f"input_{i}"] = "absence"


init_input_state()


# =========================
# Layout
# Left: all 23 feature inputs, ordered by SHAP importance
# Right: prediction results and SHAP force plot
# =========================
left_col, right_col = st.columns([1, 1.5])

with left_col:
    st.subheader("Genetic Variation Input")

    for name in display_order:
        original_index = display_to_index[name]
        st.selectbox(
            label=name,
            options=OPTION_LIST,
            key=f"input_{original_index}"
        )

    st.markdown("---")

    predict_btn = st.button(
        "Predict",
        type="primary",
        use_container_width=True
    )

    st.button(
        "Reset Inputs",
        on_click=reset_inputs,
        use_container_width=True
    )

with right_col:
    # =========================
    # Prediction and SHAP explanation
    # =========================
    if predict_btn:

        # Original dropdown selections: absence / presence, shown to users in SHAP order
        input_label_dict_display = {
            name: st.session_state[f"input_{display_to_index[name]}"]
            for name in display_order
        }

        # Numeric values by clean display name
        input_value_dict_display = {
            name: OPTION_MAP[st.session_state[f"input_{display_to_index[name]}"]]
            for name in feature_names_display
        }

        # Numeric values passed to the model, with original model feature names preserved
        input_value_dict_model = {
            display_to_model[name]: input_value_dict_display[name]
            for name in feature_names_display
        }

        X_label = pd.DataFrame([input_label_dict_display], columns=display_order)
        X = pd.DataFrame([input_value_dict_model], columns=feature_names_model)

        st.subheader("Current Input Data")

        with st.expander("View original input: absence / presence", expanded=False):
            st.dataframe(X_label, use_container_width=True)

        with st.expander("View model input values: 0 / 1", expanded=False):
            # Show a clean, readable table while keeping X itself compatible with the model.
            X_display = pd.DataFrame(
                [{name: input_value_dict_display[name] for name in display_order}],
                columns=display_order
            )
            st.dataframe(X_display, use_container_width=True)

        try:
            pred_class = clf.predict(X)[0]

            if hasattr(clf, "predict_proba"):
                pred_proba_all = clf.predict_proba(X)
                classes = list(clf.classes_)

                # By default, explain and display the probability of class 1.
                # If class 1 is not available, use the last class by default.
                if 1 in classes:
                    target_class = 1
                    target_index = classes.index(1)
                else:
                    target_class = classes[-1]
                    target_index = len(classes) - 1

                pred_proba = pred_proba_all[0][target_index]

                # Index corresponding to the predicted class, used as a fallback
                if pred_class in classes:
                    pred_index = classes.index(pred_class)
                else:
                    pred_index = int(np.argmax(pred_proba_all[0]))

            else:
                pred_proba = None
                target_index = 0
                pred_index = 0
                target_class = None

            st.subheader("Prediction Results")

            st.success(f"Predicted class: {pred_class}")

            if pred_proba is not None:
                st.success(f"Prediction probability: {pred_proba * 100:.2f}%")
            else:
                st.info("The current model does not support predict_proba, so prediction probability cannot be displayed.")

        except Exception as e:
            st.error("Model prediction failed.")
            st.exception(e)
            st.stop()


        # =========================
        # SHAP force plot
        # =========================
        st.subheader("SHAP Force Plot")

        try:
            explainer = load_explainer(clf)
            shap_values = explainer(X)

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

            # Ensure expected_value is a scalar
            expected_value = float(np.ravel(expected_value)[0])

            # Prevent old figures from remaining
            plt.close("all")

            # SHAP values follow the model feature order; use clean names for display.
            shap.force_plot(
                expected_value,
                np.round(shap_value_single, 3),
                np.round(X.iloc[0].values.astype(float), 3),
                feature_names=feature_names_display,
                figsize=(45, 4),
                matplotlib=True,
                show=False,
                text_rotation=0,
                contribution_threshold=0.03
            )

            fig = plt.gcf()

            # Save as SVG for clearer display on the webpage
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
                <div style="width:100%; overflow-x:auto; border:1px solid #e6e6e6; padding:10px;">
                    <img src="data:image/svg+xml;base64,{b64}" style="width:100%; min-width:1200px;">
                </div>
                """,
                height=360,
                scrolling=False
            )

        except Exception as e:
            st.error("Failed to generate SHAP force plot.")
            st.exception(e)
