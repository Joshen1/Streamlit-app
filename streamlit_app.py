import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from model_utils import train_models, load_data
import joblib


def main():
    st.title("California Housing — Model Trainer")

    st.markdown("This app trains a few regressors on the California housing dataset and saves the best model.")

    # Load data
    df = load_data()

    if st.checkbox("Show raw data"):
        st.write(df.head())

    st.sidebar.header("Training options")
    test_size = st.sidebar.slider("Test size (fraction)", 0.05, 0.5, 0.2, 0.05)
    random_state = st.sidebar.number_input("Random state", value=42, step=1)

    if st.sidebar.button("Train models"):
        with st.spinner("Training models..."):
            results_df, trained_models, best_model_name = train_models(df, test_size=test_size, random_state=random_state)

        st.success(f"Training complete — best model: {best_model_name}")
        st.dataframe(results_df)

        # Save best model
        best_model = trained_models[best_model_name]
        out_path = "best_model.pkl"
        joblib.dump(best_model, out_path)
        st.markdown(f"Saved best model to `{out_path}`")

        with open(out_path, "rb") as f:
            st.download_button("Download Best Model", f, file_name=out_path)


if __name__ == "__main__":
    main()
