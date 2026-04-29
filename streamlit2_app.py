import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    data = fetch_california_housing(as_frame=True)
    return data.frame

df = load_data()

TARGET = "MedHouseVal"

# -------------------------------
# Models
# -------------------------------
def get_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "LightGBM": lgb.LGBMRegressor()
    }

# -------------------------------
# Global storage
# -------------------------------
if "results" not in st.session_state:
    st.session_state.results = None
if "models" not in st.session_state:
    st.session_state.models = None
if "best_model" not in st.session_state:
    st.session_state.best_model = None
if "feature_cols" not in st.session_state:
    st.session_state.feature_cols = None

# -------------------------------
# Sidebar Navigation
# -------------------------------
page = st.sidebar.selectbox("Navigation", ["Data", "Train", "Evaluate", "Predict"])

# ===============================
# 1. DATA PAGE
# ===============================
if page == "Data":
    st.title("📊 Data Page")

    st.subheader("Dataset Preview")
    st.write(df.head())

    st.subheader("Summary Statistics")
    st.write(df.describe())

    st.subheader("Target Distribution")
    fig, ax = plt.subplots()
    df[TARGET].hist(ax=ax)
    st.pyplot(fig)

# ===============================
# 2. TRAIN PAGE
# ===============================
elif page == "Train":
    st.title("⚙️ Train Models")

    if st.button("Train All Models"):
        with st.spinner("Training in progress..."):
            X = df.drop(columns=[TARGET])
            y = df[TARGET]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            results = []
            trained_models = {}

            progress = st.progress(0)
            models = get_models()

            for i, (name, model) in enumerate(models.items()):
                cv = cross_validate(
                    model, X_train, y_train, cv=5,
                    scoring=["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]
                )

                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                rmse = mean_squared_error(y_test, preds) ** 0.5
                mae = mean_absolute_error(y_test, preds)
                r2 = r2_score(y_test, preds)

                results.append({
                    "Model": name,
                    "RMSE": rmse,
                    "MAE": mae,
                    "R2": r2,
                    "CV RMSE": np.mean(-cv["test_neg_mean_squared_error"]) ** 0.5
                })

                trained_models[name] = model
                progress.progress((i + 1) / len(models))

            results_df = pd.DataFrame(results).sort_values("RMSE")

            best_model_name = results_df.iloc[0]["Model"]
            best_model = trained_models[best_model_name]

            st.session_state.results = results_df
            st.session_state.models = trained_models
            st.session_state.best_model = best_model
            st.session_state.feature_cols = X.columns

            joblib.dump(best_model, "best_model.pkl")

        st.success(f"✅ Best Model Saved: {best_model_name}")
        st.dataframe(results_df)

# ===============================
# 3. EVALUATE PAGE
# ===============================
elif page == "Evaluate":
    st.title("📈 Evaluate Models")

    if st.session_state.results is None:
        st.warning("Train models first.")
    else:
        results_df = st.session_state.results

        st.subheader("Leaderboard")
        st.dataframe(results_df.style.highlight_min(subset=["RMSE"], color="lightgreen"))

        best_model = st.session_state.best_model
        X = df.drop(columns=[TARGET])
        y = df[TARGET]

        preds = best_model.predict(X)

        # Prediction Plot
        st.subheader("Prediction vs Actual")
        fig, ax = plt.subplots()
        ax.scatter(y, preds, alpha=0.3)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        st.pyplot(fig)

        # Residual Plot
        st.subheader("Residual Plot")
        residuals = y - preds
        fig, ax = plt.subplots()
        ax.scatter(preds, residuals, alpha=0.3)
        ax.axhline(0)
        st.pyplot(fig)

        # Feature Importance
        if hasattr(best_model, "feature_importances_"):
            st.subheader("Feature Importance")
            importance = best_model.feature_importances_
            features = st.session_state.feature_cols

            imp_df = pd.DataFrame({
                "Feature": features,
                "Importance": importance
            }).sort_values("Importance", ascending=False)

            st.bar_chart(imp_df.set_index("Feature"))

# ===============================
# 4. PREDICT PAGE
# ===============================
elif page == "Predict":
    st.title("🔮 Prediction")

    if st.session_state.best_model is None:
        st.warning("Train a model first.")
    else:
        model = st.session_state.best_model
        feature_cols = st.session_state.feature_cols

        st.subheader("Single Prediction")

        inputs = []
        for col in feature_cols:
            val = st.number_input(col, value=0.0)
            inputs.append(val)

        if st.button("Predict"):
            pred = model.predict([inputs])
            st.success(f"Prediction: {pred[0]}")

        # -----------------------
        # Batch Prediction
        # -----------------------
        st.subheader("Batch Prediction (CSV Upload)")
        file = st.file_uploader("Upload CSV", type=["csv"])

        if file:
            try:
                batch_df = pd.read_csv(file)

                if not all(col in batch_df.columns for col in feature_cols):
                    st.error("CSV missing required columns.")
                else:
                    preds = model.predict(batch_df[feature_cols])
                    batch_df["Prediction"] = preds

                    st.write(batch_df.head())

                    csv = batch_df.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        "Download Predictions",
                        csv,
                        "predictions.csv",
                        "text/csv"
                    )

            except Exception as e:
                st.error(f"Error: {e}")
                