import pandas as pd
from datetime import datetime


def monitor(latest_predictions: pd.DataFrame) -> pd.DataFrame:
    # Start filling function
    monitoring_df = pd.DataFrame({
        "predictions_time": [latest_predictions["predictions_time"].max()],
        "predictions": [latest_predictions["predictions"].mean()]
    })
    # End filling function
    return monitoring_df
