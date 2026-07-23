import os
import time

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier


def train_model(features: pd.DataFrame, model_registry_folder: str) -> None:
    target = 'Ba_avg'
    df_x = features.drop(columns=[target])
    y = features[target]
    with mlflow.start_run():
        # log_models set to False because it doesn't work so we will log manually
        mlflow.xgboost.autolog(log_models=False)
        model = XGBClassifier(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                n_jobs=1,
                random_state=42,
                eval_metric="logloss",
                )

        model.fit(df_x, y)
        mlflow.xgboost.log_model(
            sk_model=model,
            name="xgboost_model",
            registered_model_name="resgistred_xgboost_model",
        )
    time_str = time.strftime('%Y%m%d-%H%M%S')
    joblib.dump(model, os.path.join(model_registry_folder, time_str + '.joblib'))


def predict(features: pd.DataFrame, model_path: str) -> pd.DataFrame:
    model = joblib.load(model_path)
    features['predictions'] = model.predict(features)
    return features
