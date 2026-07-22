import os
import time

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def train_model(features: pd.DataFrame, model_registry_folder: str) -> None:
    target = 'Ba_avg'
    df_x = features.drop(columns=[target])
    y = features[target]
    time_str = time.strftime('%Y%m%d-%H%M%S')
    with mlflow.start_run() as run:
        mlflow.sklearn.autolog()
        model = RandomForestRegressor(n_estimators=1, max_depth=10, n_jobs=1)
        model.fit(df_x, y)
        mlflow.sklearn.log_model(
            sk_model=model,
            name="leia",
            input_example=df_x.iloc[:1,:],
            registered_model_name="leiamodel",
        )


def predict(features: pd.DataFrame, model_path: str) -> pd.DataFrame:
    model = joblib.load(model_path)
    features['predictions'] = model.predict(features)
    return features
