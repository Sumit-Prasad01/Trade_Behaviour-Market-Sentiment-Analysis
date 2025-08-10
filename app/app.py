import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Trade & Sentiment Prediction", layout="wide")
sns.set_style("whitegrid")

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_model(model_path):
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ==========================
# SIDEBAR
# ==========================
st.sidebar.header("Model & Data Selection")

model_type = st.sidebar.radio("Select Prediction Type", ["Classification", "Regression"])
model_file = st.sidebar.file_uploader("Upload Trained Model (.pkl)", type=["pkl"])
uploaded_data = st.sidebar.file_uploader("Upload Data for Prediction (.csv)", type=["csv"])

if model_file:
    model = load_model(model_file)
else:
    model = None
    st.sidebar.warning("Please upload a trained model file.")

# ==========================
# MAIN APP
# ==========================
st.title("📈 Trade & Sentiment Predictions")

if uploaded_data is not None:
    try:
        df = pd.read_csv(uploaded_data)
        st.write("### Uploaded Data Preview")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error reading uploaded data: {e}")
        df = None
else:
    df = None
    st.info("Please upload a CSV file containing features for prediction.")

# ==========================
# PREDICTION
# ==========================
if st.button("Run Predictions") and model is not None and df is not None:
    try:
        predictions = model.predict(df)

        if model_type == "Classification":
            st.write("### Predicted Classes")
            st.dataframe(pd.DataFrame({"Prediction": predictions}))

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(df)[:, 1]
                st.write("### Prediction Probabilities")
                st.dataframe(pd.DataFrame({"Probability": proba}))

                # Probability distribution
                fig, ax = plt.subplots()
                sns.histplot(proba, bins=20, kde=True, ax=ax)
                ax.set_title("Probability Distribution")
                st.pyplot(fig)

        elif model_type == "Regression":
            st.write("### Predicted Values")
            st.dataframe(pd.DataFrame({"Prediction": predictions}))

            # Plot predicted values
            fig, ax = plt.subplots()
            sns.histplot(predictions, bins=20, kde=True, ax=ax)
            ax.set_title("Predicted Value Distribution")
            st.pyplot(fig)

        st.success("Predictions completed successfully!")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
