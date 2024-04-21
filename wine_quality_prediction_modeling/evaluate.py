"""Evaluate model."""

import json
from pathlib import Path
import joblib
from loguru import logger
import numpy as np

import pandas as pd
from sklearn import metrics


def main():
    """Main."""

    logger.info("Evaluating the model...")
    prepared_data_dir = Path(__file__).parent.parent / "data/prepared"
    test_csv_path = prepared_data_dir / "test.csv"
    test_data = pd.read_csv(test_csv_path)
    X_test = test_data.drop("quality", axis=1)
    labels = test_data["quality"]

    model_dir = Path(__file__).parent.parent / "model"
    model_file = model_dir / "wine_quality_predictor.joblib"
    logger.info(f"Loading the model from {model_file}...")
    model = joblib.load(model_file)
    logger.info("Performing predictions...")
    predictions = model.predict(X_test)

    # Root Mean Squared Error
    logger.info("Computing RMSE...")
    rmse = np.sqrt(metrics.mean_squared_error(labels, predictions))
    # Mean Absolute Error
    logger.info("Computing MAE...")
    mae = metrics.mean_absolute_error(labels, predictions)
    computed_metrics = {
        "Root Mean Squared Error": rmse,
        "Mean Absolute Error": mae,
    }

    # Save metrics
    metrics_path = Path(__file__).parent.parent / "metrics/metrics.json"
    logger.info(f"Saving metrics to {metrics_path}...")
    metrics_path.write_text(json.dumps(computed_metrics))


if __name__ == "__main__":
    main()
