"""
Leakage-free feature generation for walk-forward validation.

The original StockDataLoader computes indicators over the full dataset
before splitting — this leaks future data into training features via
rolling windows. This module fixes that by computing indicators only
on data up to a given cutoff date.
"""
import numpy as np
import pandas as pd
from features.indicators import add_indicators


def get_features_at(df: pd.DataFrame, cutoff_date: str):
    """
    Compute features using only data up to and including cutoff_date.

    Args:
        df:          Raw OHLCV DataFrame with DatetimeIndex.
        cutoff_date: ISO date string e.g. '2023-06-01'. Only rows on or
                     before this date are used to compute indicators.

    Returns:
        X     (ndarray):          Feature matrix, shape (n_samples, n_features)
        y     (ndarray):          Target — Close prices, shape (n_samples,)
        dates (DatetimeIndex):    Corresponding dates for X and y
    """
    # Slice to cutoff — no future data leaks into rolling calculations
    sliced = df.loc[:cutoff_date].copy()

    # Compute indicators on the sliced data only
    sliced = add_indicators(sliced)

    # Drop rows with NaN (created by rolling windows at the start)
    sliced = sliced.dropna()

    # Target
    y = sliced['Close'].values

    # Features: everything except raw OHLCV
    exclude = {'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'}
    feature_cols = [c for c in sliced.columns if c not in exclude]
    X = sliced[feature_cols].values

    return X, y, sliced.index


if __name__ == "__main__":
    from utils.data_loader import StockDataLoader

    loader = StockDataLoader("AAPL", "2020-01-01", "2024-01-01")
    loader.download_data()

    X_wf, y_wf, dates_wf = get_features_at(loader.raw_data, "2022-01-01")

    # Verify no future leak
    assert dates_wf[-1] <= pd.Timestamp("2022-01-01"), "Cutoff not respected!"

    # Verify shape matches original pipeline
    loader.add_technical_indicators()
    X_orig, y_orig, _ = loader.prepare_features()
    assert X_wf.shape[1] == X_orig.shape[1], \
        f"Feature count mismatch: {X_wf.shape[1]} vs {X_orig.shape[1]}"

    print(f"walk_forward shape : {X_wf.shape}")
    print(f"original shape     : {X_orig.shape}")
    print(f"Last date in slice : {dates_wf[-1].date()}  (cutoff: 2022-01-01)")
    print("features/walk_forward.py: OK")
