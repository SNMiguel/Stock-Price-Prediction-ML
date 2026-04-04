"""
Merges daily sentiment scores onto a feature matrix.
Called by walk_forward_trainer and daily_job after features are computed.
"""
import pandas as pd
import numpy as np


def merge_sentiment(feature_df: pd.DataFrame,
                    sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join sentiment scores onto the feature matrix by date index.

    Args:
        feature_df:   DataFrame with DatetimeIndex and feature columns.
        sentiment_df: DataFrame with DatetimeIndex and a 'score' column
                      (as returned by Database.get_sentiment()).

    Returns:
        feature_df copy with one extra 'sentiment' column.
        Missing sentiment days (weekends, holidays, no news) are
        forward-filled then back-filled; remaining NaN filled with 0.0.
    """
    df = feature_df.copy()

    if sentiment_df.empty:
        df['sentiment'] = 0.0
        return df

    # Rename to avoid column clash if already present
    scores = sentiment_df[['score']].rename(columns={'score': 'sentiment'})

    df = df.join(scores, how='left')
    df['sentiment'] = df['sentiment'].ffill().bfill().fillna(0.0)

    return df


if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    # Simulate a feature matrix (5 days)
    dates = pd.bdate_range("2023-01-02", periods=5)
    feature_df = pd.DataFrame(
        np.random.randn(5, 3),
        index=dates,
        columns=['MA_5', 'RSI', 'MACD']
    )

    # Simulate sentiment (only 3 of the 5 days have scores)
    sentiment_df = pd.DataFrame(
        {'score': [0.12, -0.05, 0.30]},
        index=dates[[0, 2, 4]]
    )

    result = merge_sentiment(feature_df, sentiment_df)

    assert 'sentiment' in result.columns
    assert result['sentiment'].isna().sum() == 0, "NaN values remain after fill"
    assert result.shape == (5, 4)

    print(result)
    print("features/sentiment_features.py: OK")
