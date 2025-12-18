import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import shap
import matplotlib
# 关键：兼容Streamlit Cloud无GUI环境
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# ===================== 0. Global Configuration (English) =====================
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# Page Configuration
st.set_page_config(
    page_title="Hypoproteinemia Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== 1. Path Configuration (Cloud-Compatible) =====================
# 模型/验证集路径（与app.py同目录）
MODEL_PATH = "hypoproteinemia_model_final.pkl"
VAL_DATA_PATH = "validation_data.xlsx"

# ===================== 2. Model Loading (Error-Resistant) =====================
@st.cache_resource
def load_model():
    """Load model with validation for core components"""
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found! Path: {MODEL_PATH}")
        st.stop()

    try:
        save_dict = joblib.load(MODEL_PATH)
        # Core components (required)
        model = save_dict.get('model')
        scaler = save_dict.get('scaler')
        encoders = save_dict.get('encoders', {})
        median_dict = save_dict.get('median_dict', {})
        mode_dict = save_dict.get('mode_dict', {})
        # SHAP components (optional)
        explainer = save_dict.get('shap_explainer')
        shap_values_val = save_dict.get('shap_values_val')

        # Validate core components
        if model is None or scaler is None:
            st.error("❌ Model corrupted! Missing core components (model/scaler)")
            st.stop()

        return model, scaler, encoders, median_dict, mode_dict, explainer, shap_values_val
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.error("Re-run training script to regenerate model file!")
        st.stop()

# Load model components
model, scaler, encoders, median_dict, mode_dict, explainer, shap_values_val = load_model()

# ===================== 3. Validation Data Loading =====================
@st.cache_data
def load_val_data():
    """Load and preprocess validation data"""
    if not os.path.exists(VAL_DATA_PATH):
        st.error(f"❌ Validation data not found! Path: {VAL_DATA_PATH}")
        st.stop()

    # Read raw data
    val_df = pd.read_excel(VAL_DATA_PATH, header=0, engine='openpyxl')
    # Split features and target (adjust column indices based on your data)
    X_val = val_df.iloc[:, 1:].copy()  # Features (adjust columns as needed)
    y_val = val_df.iloc[:, 0].copy()   # Target (hypoproteinemia label)

    # Classify feature types
    numeric_cols = X_val.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X_val.select_dtypes(include=['object', 'category']).columns

    # Preprocessing (match training logic)
    # 1. Fill missing values
    for col in numeric_cols:
        if col in median_dict:
            X_val[col].fillna(median_dict[col], inplace=True)
    for col in categorical_cols:
        if col in mode_dict:
            X_val[col].fillna(mode_dict[col], inplace=True)

    # 2. Encode categorical features
    for col in categorical_cols:
        if col in encoders:
            try:
                X_val[col] = encoders[col].transform(X_val[col].astype(str))
            except:
                X_val[col] = 0  # Default for unseen categories

    # 3. Standardize numeric features
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])

    return val_df, X_val, y_val, numeric_cols, categorical_cols

# Load validation data
val_df_ori, X_val, y_val, numeric_cols, categorical_cols = load_val_data()

# ===================== 4. Sidebar: Function Selection (Only 2 Options) =====================
st.sidebar.title("Function Menu")
function_choice = st.sidebar.radio(
    "Select Function",
    ["🔮 Single Sample Prediction", "📈 Interpretability Analysis"]  # Only 2 functions
)

# ===================== 5. Single Sample Prediction (Core Function) =====================
if function_choice == "🔮 Single Sample Prediction":
    st.title("Hypoproteinemia - Single Sample Prediction")
    st.markdown("### Enter Patient Features")

    # Build input form
    input_data = {}
    col1, col2 = st.columns(2)

    # Numeric features input
    with col1:
        st.subheader("Numeric Features")
        for col in numeric_cols:
            # Get safe stats for input range
            min_val = float(val_df_ori[col].min()) if not val_df_ori[col].isna().all() else 0.0
            max_val = float(val_df_ori[col].max()) if not val_df_ori[col].isna().all() else 100.0
            mean_val = float(val_df_ori[col].median()) if not val_df_ori[col].isna().all() else 50.0

            input_data[col] = st.number_input(
                f"{col}",
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=0.1
            )

    # Categorical features input
    with col2:
        st.subheader("Categorical Features")
        for col in categorical_cols:
            unique_vals = val_df_ori[col].dropna().unique() if col in val_df_ori.columns else []
            if len(unique_vals) == 0:
                unique_vals = ["Unknown"]
            input_data[col] = st.selectbox(f"{col}", unique_vals)

    # Prediction button
    if st.button("🚀 Start Prediction"):
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Preprocessing
        # Fill missing values
        for col in numeric_cols:
            input_df[col].fillna(0.0, inplace=True)
        for col in categorical_cols:
            input_df[col].fillna(unique_vals[0], inplace=True)

        # Encode categorical features
        for col in categorical_cols:
            if col in encoders:
                try:
                    input_df[col] = encoders[col].transform(input_df[col].astype(str))
                except:
                    input_df[col] = 0

        # Standardize numeric features
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        # Prediction
        pred_proba = model.predict_proba(input_df)[0, 1]
        pred_label = 1 if pred_proba >= 0.5 else 0
        pred_text = "Hypoproteinemia" if pred_label == 1 else "No Hypoproteinemia"

        # Show results
        st.markdown("### Prediction Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Class", pred_text)
            st.metric("Hypoproteinemia Probability", f"{pred_proba:.4f}")
        with col2:
            # SHAP explanation (single sample)
            if explainer is not None:
                st.markdown("#### Feature Impact Explanation (SHAP)")
                shap_val = explainer.shap_values(input_df)
                fig, ax = plt.subplots(figsize=(10, 4))
                shap.force_plot(
                    explainer.expected_value,
                    shap_val[0],
                    input_data,
                    matplotlib=True,
                    show=False,
                    figsize=(10, 4),
                    ax=ax
                )
                st.pyplot(fig)
            else:
                st.warning("⚠️ SHAP explainer not loaded! Re-train model with SHAP components.")

# ===================== 6. Interpretability Analysis (SHAP) =====================
elif function_choice == "📈 Interpretability Analysis":
    st.title("Model Interpretability Analysis (SHAP)")

    # Check SHAP components
    if explainer is None or shap_values_val is None:
        st.warning("⚠️ SHAP components not loaded! Re-train model and save SHAP explainer/values.")
    else:
        tab1, tab2, tab3 = st.tabs(["📊 SHAP Summary Plot", "🔍 Feature Dependence Plot", "🧬 Single Sample Explanation"])

        # Tab1: SHAP Summary Plot
        with tab1:
            st.markdown("### SHAP Summary Plot (Validation Set)")
            st.markdown("""
            - Y-axis: Feature importance (top = more impactful)
            - X-axis: SHAP value (positive = increase risk, negative = decrease risk)
            - Color: Feature value (red = high, blue = low)
            """)
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.summary_plot(
                shap_values_val, X_val,
                feature_names=X_val.columns,
                plot_type="dot",
                show=False,
                ax=ax,
                cmap=plt.get_cmap("coolwarm")
            )
            st.pyplot(fig)

        # Tab2: Feature Dependence Plot
        with tab2:
            st.markdown("### Feature Dependence Plot (Top 5 Features)")
            # Calculate top 5 features by SHAP importance
            shap_importance = np.abs(shap_values_val).mean(axis=0)
            top5_feat = X_val.columns[np.argsort(shap_importance)[-5:]][::-1]
            selected_feat = st.selectbox("Select Feature", top5_feat)

            fig, ax = plt.subplots(figsize=(8, 6))
            shap.dependence_plot(
                selected_feat,
                shap_values_val,
                X_val,
                feature_names=X_val.columns,
                show=False,
                ax=ax,
                alpha=0.6,
                dot_size=20
            )
            ax.set_title(f"Feature Dependence Plot - {selected_feat}")
            st.pyplot(fig)

        # Tab3: Single Sample Explanation (Validation Set)
        with tab3:
            st.markdown("### Single Sample SHAP Explanation (Validation Set)")
            # Select sample index
            sample_idx = st.slider("Select Validation Sample Index", 0, len(X_val) - 1, 100)
            # Show sample features
            st.markdown("#### Sample Features")
            sample_data = val_df_ori.iloc[sample_idx]
            st.write(sample_data)

            # Show SHAP force plot
            st.markdown("#### Feature Impact Explanation")
            fig, ax = plt.subplots(figsize=(12, 4))
            shap.force_plot(
                explainer.expected_value,
                shap_values_val[sample_idx],
                X_val.iloc[sample_idx],
                feature_names=X_val.columns,
                matplotlib=True,
                show=False,
                figsize=(12, 4),
                ax=ax
            )
            ax.set_title(f"Sample {sample_idx} - True Label: {'Hypoproteinemia' if y_val.iloc[sample_idx] == 1 else 'No Hypoproteinemia'}")
            st.pyplot(fig)

# ===================== 7. Footer =====================
st.markdown("---")
st.markdown("© 2025 Hypoproteinemia Prediction Model | Streamlit Web App | LightGBM + SHAP Interpretability")