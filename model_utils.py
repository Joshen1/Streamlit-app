import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from sklearn.datasets import fetch_california_housing


def load_data(as_frame=False):
    # return a DataFrame similar to the notebook
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["MedHouseVal"] = data.target
    return df


def get_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }


def train_models(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Train models and return results, trained model dict, and best model name.

    Returns:
        results_df (pd.DataFrame): Metrics per model sorted by RMSE (ascending)
        trained_models (dict): name -> fitted estimator
        best_model_name (str)
    """
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    models = get_models()
    results = []
    trained_models = {}

    for name, model in models.items():
        # cross-validate for quick diagnostics (not used in final ranking)
        try:
            _ = cross_validate(model, X_train, y_train, cv=5,
                               scoring=["neg_mean_squared_error", "neg_mean_absolute_error", "r2"])
        except Exception:
            # some models may fail CV with default settings; ignore
            pass

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = mean_squared_error(y_test, preds, squared=False)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results.append({
            "Model": name,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        })

        trained_models[name] = model

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)

    best_model_name = results_df.iloc[0]["Model"]

    return results_df, trained_models, best_model_name
