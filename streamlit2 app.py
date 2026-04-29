import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    data = fetch_california_housing(as_frame=True)
    return data.frame

df = load_data()

# -------------------------------
# Models
# -------------------------------
def get_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "LightGBM": lgb.LGBMRegressor(
        )
    }

# -------------------------------
# Train Models
# -------------------------------
@st.cache_resource
def train_models(df):
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    models = get_models()
    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = mean_squared_error(y_test, preds) ** 0.5
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results.append({
            "Model": name,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        })

        trained_models[name] = model

    results_df = pd.DataFrame(results).sort_values("RMSE")

    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    joblib.dump(best_model, "best_model.pkl")
    joblib.dump(X.columns.tolist(), "features.pkl")

    return results_df, best_model_name


# -------------------------------
# UI
# -------------------------------
st.title("🏠 California Housing ML App")

menu = st.sidebar.selectbox("Menu", ["Dataset", "Train Models", "Predict"])

# -------------------------------
# Dataset View
# -------------------------------
if menu == "Dataset":
    st.subheader("Dataset Preview")
    st.write(df.head())
    st.write(f"Shape: {df.shape}")

# -------------------------------
# Train Models
# -------------------------------
elif menu == "Train Models":
    st.subheader("Training Models")

    if st.button("Train"):
        with st.spinner("Training models..."):
            results_df, best_model_name = train_models(df)


        st.subheader("Model Performance")
        st.dataframe(results_df)

        st.write(f"🏆 Best Model: **{best_model_name}**")

# -------------------------------
# Prediction
# -------------------------------
elif menu == "Predict":
    st.subheader("Make Prediction")

    try:
        model = joblib.load("best_model.pkl")
        feature_cols = joblib.load("features.pkl")

        input_data = []

        for col in feature_cols:
            val = st.number_input(col, value=0.0)
            input_data.append(val)

        if st.button("Predict"):
            input_array = np.array([input_data])
            prediction = model.predict(input_array)

            st.success(f"Predicted House Value: {prediction[0]}")

    except:
        st.warning("⚠️ Please train the model first.")