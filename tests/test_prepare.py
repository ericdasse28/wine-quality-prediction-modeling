"""Test data preparation."""


import pandas as pd
from wine_quality_prediction_modeling import prepare


def test_get_features():
    """Test get_features."""

    correlations = pd.Series(
        {
            "fixed acidity": 0.124052,
            "volatile acidity": -0.390558,
            "citric acid": 0.226373,
            "residual sugar": 0.013732,
            "chlorides": -0.128907,
            "free sulfur dioxide": -0.050656,
            "total sulfur dioxide": -0.185100,
            "density": -0.174919,
            "pH": -0.057731,
            "sulphates": 0.251397,
            "alcohol": 0.476166,
        }
    )

    actual_features = prepare.get_features_from_correlations(
        correlations=correlations, correlation_threshold=0.05
    )

    expected_features = [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
    ]
    assert expected_features == actual_features
