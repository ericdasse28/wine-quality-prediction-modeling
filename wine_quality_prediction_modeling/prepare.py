"""Data preparation for training related functions."""

import pandas as pd


def get_features(correlations: pd.Series, correlation_threshold: float):
    """Output features whose correlation is above a threshold
    value."""

    abs_corrs = correlations.abs()
    high_correlations = list(
        abs_corrs[abs_corrs > correlation_threshold].index.values,
    )

    return high_correlations
