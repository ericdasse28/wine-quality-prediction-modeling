"""Data preparation module."""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from loguru import logger


def generate_train_and_test_data(
    X: pd.DataFrame,
    y: pd.Series,
    prepared_data_dir: Path,
):
    """Generate train and test data."""

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    # Save training and test data
    train_data.to_csv(prepared_data_dir / "train.csv", index=False)
    test_data.to_csv(prepared_data_dir / "test.csv", index=False)


def get_features_from_correlations(
    correlations: pd.Series, correlation_threshold: float
) -> list[str]:
    """Get features from correlations."""

    abs_corrs = correlations.abs()
    high_correlations = list(
        abs_corrs[abs_corrs > correlation_threshold].index.values,
    )

    return high_correlations


def get_features(
    raw_df: pd.DataFrame,
    correlation_threshold: float,
) -> list[str]:
    """Output features whose correlation is above a threshold
    value."""

    correlations = raw_df.corr()["quality"].drop("quality")

    return get_features_from_correlations(
        correlations, correlation_threshold=correlation_threshold
    )


def prepare_data(raw_data_file_path: Path, prepared_data_dir_path: Path):
    """Prepare data."""

    raw_data = pd.read_csv(raw_data_file_path, sep=";")

    features = get_features(raw_data, correlation_threshold=0.05)

    X = raw_data[features]
    y = raw_data["quality"]

    # Generate CSV files for training and test data
    generate_train_and_test_data(
        X,
        y,
        prepared_data_dir=prepared_data_dir_path,
    )


def main():
    """Main."""

    raw_data_file_path = (
        Path(__file__).parent.parent
        / "data/raw/winequality-\
red.csv"
    )
    prepared_data_dir_path = Path(__file__).parent.parent / "data/prepared"
    # Create prepared directory if it does not exist already
    prepared_data_dir_path.mkdir(exist_ok=True)

    logger.info(
        f"Running data preparation on raw data at {raw_data_file_path}...",
    )
    prepare_data(raw_data_file_path, prepared_data_dir_path)


if __name__ == "__main__":
    main()
