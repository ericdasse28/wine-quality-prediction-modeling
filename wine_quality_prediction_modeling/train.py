"""Training module."""

from pathlib import Path
import joblib

import pandas as pd
from sklearn.linear_model import LinearRegression


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
    trained_model = train(training_data_path)

    # Save model to model dir
    model_dir = Path(__file__).parent.parent / "model"
    model_dir.mkdir(exist_ok=True)
    model_dump = model_dir / "wine_quality_model.joblib"
    joblib.dump(trained_model, model_dump)


if __name__ == "__main__":
    main()
