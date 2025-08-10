import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

# ==========================
# CONFIG
# ==========================
st.set_page_config(page_title="Trade & Sentiment Modeling", layout="wide")
sns.set_style("whitegrid")

# ==========================
# LOAD DEFAULT DATA
# ==========================
@st.cache_data
def load_default_data():
    return pd.read_csv("data/processed/merged_trades_sentiment.csv", parse_dates=['date'])

df = load_default_data()

# ==========================
# SIDEBAR INPUTS
# ==========================
st.sidebar.header("Modeling Settings")
uploaded_file = st.sidebar.file_uploader("Upload your CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=['date'])

target_type = st.sidebar.radio("Target Type", ["Classification", "Regression"])
target_column = st.sidebar.selectbox("Select Target Column", options=df.columns)
test_size = st.sidebar.slider("Test Size %", 10, 50, 20) / 100
random_state = st.sidebar.number_input("Random State", 0, 999, 42)

# Feature selection
features = st.sidebar.multiselect("Select Features", options=[c for c in df.columns if c != target_column])

# Model choice
if target_type == "Classification":
    model_choice = st.sidebar.selectbox("Select Model", ["Logistic Regression", "XGBoost", "LightGBM"])
else:
    model_choice = st.sidebar.selectbox("Select Model", ["Linear Regression", "XGBoost", "LightGBM"])

# ==========================
# TRAINING
# ==========================
if st.sidebar.button("Train Model"):
    X = df[features]
    y = df[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=False)

    # Scaling for certain models
    scaler = None
    if model_choice == "Logistic Regression" or model_choice == "Linear Regression":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Model initialization
    if target_type == "Classification":
        if model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_choice == "XGBoost":
            model = XGBClassifier(n_estimators=200, learning_rate=0.05, eval_metric='logloss', random_state=random_state)
        elif model_choice == "LightGBM":
            model = LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=random_state)
    else:
        if model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "XGBoost":
            model = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=random_state)
        elif model_choice == "LightGBM":
            model = LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=random_state)

    # Fit model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # ==========================
    # EVALUATION
    # ==========================
    st.subheader("Model Performance")

    if target_type == "Classification":
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        st.write(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
        st.write(f"Recall: {recall_score(y_test, y_pred, zero_division=0):.4f}")
        st.write(f"F1 Score: {f1_score(y_test, y_pred, zero_division=0):.4f}")
    else:
        st.write(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")
        st.write(f"R² Score: {r2_score(y_test, y_pred):.4f}")

    # ==========================
    # FEATURE IMPORTANCE
    # ==========================
    if hasattr(model, "feature_importances_"):
        st.subheader("Feature Importance")
        fi_df = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
        fig, ax = plt.subplots()
        sns.barplot(x="Importance", y="Feature", data=fi_df, ax=ax)
        st.pyplot(fig)

    # ==========================
    # SHAP EXPLAINABILITY
    # ==========================
    try:
        st.subheader("SHAP Summary Plot")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_test, feature_names=features, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.info(f"SHAP not available for {model_choice}: {e}")

    # ==========================
    # SAVE MODEL
    # ==========================
    if st.button("Save Model"):
        joblib.dump(model, f"{model_choice.replace(' ', '_').lower()}_{target_type.lower()}.pkl")
        st.success("Model saved successfully!")
