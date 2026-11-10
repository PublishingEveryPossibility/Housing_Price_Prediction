from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_PATH = DATA_DIR / "housing.csv"


def fetch_housing_data() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"

    if not DATA_PATH.exists():
        import urllib.request

        urllib.request.urlretrieve(url, DATA_PATH)

    return pd.read_csv(DATA_PATH)


def build_model() -> Pipeline:
    numeric_features = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
    ]
    categorical_features = ["ocean_proximity"]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(n_estimators=200, random_state=42),
            ),
        ]
    )
    return model


def main() -> dict[str, float]:
    df = fetch_housing_data()
    target = "median_house_value"
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_model()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"MAE: ${mae:,.0f}")
    print(f"RMSE: ${rmse:,.0f}")
    print(f"R^2: {r2:.3f}")

    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


if __name__ == "__main__":
    main()
