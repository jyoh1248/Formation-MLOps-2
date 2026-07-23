import pandas as pd
import mlflow

def monitor(latest_predictions: pd.DataFrame) -> pd.DataFrame:
    monitoring_df = latest_predictions.groupby('predictions_time').agg({'predictions': 'mean'}).reset_index()
    return monitoring_df

def shadow_predict_with_io(features_path,  model_path, ):
    features = pd.read_parquet(features_path)
    # Load model
    model = mlflow.pyfunc.load_model(model_path)
    # Predict with shadow model
    predictions = model.predict(features)
    # Compute metrics
    monitoring_df = monitor(predictions)
    # Send metrics to postgresql
    engine = create_engine(db_con_str)
    db_conn = engine.connect()
    monitoring_df.to_sql(shadow_monitoring_table_name, con=db_conn, if_exists='append', index=False)
    db_conn.close()


    