"""Training module."""

from pathlib import Path
import joblib

from loguru import logger
import pandas as pd
from sklearn.linear_model import LinearRegression
from dvclive import Live


def train(training_data_path: Path) -> LinearRegression:
    """Model training."""

    # Get the training data
    training_data = pd.read_csv(training_data_path)
    X_train = training_data.drop("quality", axis=1)
    y_train = training_data["quality"]

    # Train the model
    regressor = LinearRegression()
    trained_model = regressor.fit(X_train, y_train)

    return trained_model


def main():
    """Main."""

    prepared_data_path = Path(__file__).parent.parent / "data/prepared"
    training_data_path = prepared_data_path / "train.csv"

    with Live() as live:
        logger.info("Training the wine quality prediction model...")
        trained_model = train(training_data_path)

        # Save model to model dir
        model_dir = Path(__file__).parent.parent / "model"
        model_dir.mkdir(exist_ok=True)
        model_dump = model_dir / "wine_quality_model.joblib"
        logger.info(f"Saving model to {model_dump}...")
        joblib.dump(trained_model, model_dump)

        live.log_artifact(str(model_dump), type="model")

    logger.success("Training successful!")


if __name__ == "__main__":
    main()
