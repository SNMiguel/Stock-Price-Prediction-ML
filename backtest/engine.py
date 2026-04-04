"""
Event-driven backtester.

Iterates over historical dates chronologically. At each step:
  1. Compute features using only data up to that date (no leakage)
  2. Generate a signal from the loaded model
  3. Simulate fill at the NEXT day's open price
  4. Apply commission
  5. Track portfolio value, cash, and position

Returns an equity curve DataFrame that backtest/report.py uses
to compute financial metrics.
"""
import numpy as np
import pandas as pd

from features.walk_forward import get_features_at
from features.sentiment_features import merge_sentiment


class BacktestEngine:
    """Chronological event-driven backtester."""

    def __init__(self, commission_per_share: float = 0.01,
                 initial_capital: float = 100_000.0):
        """
        Args:
            commission_per_share: Cost per share traded (default $0.01).
            initial_capital:      Starting portfolio cash (default $100,000).
        """
        self.commission_per_share = commission_per_share
        self.initial_capital      = initial_capital

    def run(self, df: pd.DataFrame,
            sentiment_df: pd.DataFrame,
            model,
            signal_gen,
            sizer) -> pd.DataFrame:
        """
        Run the full backtest.

        Args:
            df:           Raw OHLCV DataFrame with DatetimeIndex.
                          Must cover at least 60 rows (rolling window warmup).
            sentiment_df: Sentiment scores DataFrame (date index, 'score' col).
                          Can be empty.
            model:        Any object with .predict(X) -> ndarray.
                          If it has .get_confidence(X), that is used;
                          otherwise confidence defaults to 0.7.
            signal_gen:   SignalGenerator instance.
            sizer:        PositionSizer instance.

        Returns:
            DataFrame with columns:
                date, portfolio_value, cash, position_qty,
                trade_side, trade_qty, trade_price
            One row per trading day in the backtest window.
        """
        dates   = df.index
        n       = len(dates)
        warmup  = 55   # minimum rows needed for all indicators (MA_50 = 50)

        cash     = self.initial_capital
        position = 0       # shares held (positive = long, negative = short)
        entry_px = 0.0     # price at which current position was opened

        records = []

        for i in range(warmup, n - 1):
            today     = dates[i]
            tomorrow  = dates[i + 1]
            today_str = today.strftime('%Y-%m-%d')

            # Features up to and including today (no future leakage)
            X, y, feat_dates = get_features_at(df, today_str)

            if len(X) == 0:
                continue

            # Merge sentiment
            n_features = X.shape[1]
            feat_df    = pd.DataFrame(
                X, index=feat_dates,
                columns=[f'f{j}' for j in range(n_features)]
            )
            feat_df = merge_sentiment(feat_df, sentiment_df)
            X_merged = feat_df.values

            # Predict using last row (today)
            X_today = X_merged[-1:].reshape(1, -1)
            predicted = float(model.predict(X_today)[0])

            # Confidence
            if hasattr(model, 'get_confidence'):
                confidence = model.get_confidence(X_today)
            else:
                confidence = 0.7

            current_price = float(df.loc[today, 'Close'])
            fill_price    = float(df.loc[tomorrow, 'Open'])

            signal = signal_gen.generate(current_price, predicted, confidence)
            action = signal['signal']

            trade_side = ''
            trade_qty  = 0
            trade_px   = 0.0

            # --- Execute signal ---
            if action == 'BUY' and position == 0:
                atr      = sizer.calculate_atr(df.loc[:today])
                pv       = cash + position * current_price
                qty      = sizer.size(pv, current_price, atr)

                if qty > 0:
                    cost       = qty * fill_price
                    commission = qty * self.commission_per_share

                    if cost + commission <= cash:
                        cash      -= (cost + commission)
                        position  += qty
                        entry_px   = fill_price
                        trade_side = 'BUY'
                        trade_qty  = qty
                        trade_px   = fill_price

            elif action == 'SELL' and position > 0:
                qty        = position
                proceeds   = qty * fill_price
                commission = qty * self.commission_per_share

                cash      += proceeds - commission
                position   = 0
                entry_px   = 0.0
                trade_side = 'SELL'
                trade_qty  = qty
                trade_px   = fill_price

            portfolio_value = cash + position * fill_price

            records.append({
                'date':            tomorrow,
                'portfolio_value': round(portfolio_value, 2),
                'cash':            round(cash, 2),
                'position_qty':    position,
                'trade_side':      trade_side,
                'trade_qty':       trade_qty,
                'trade_price':     round(trade_px, 4),
            })

        equity_curve = pd.DataFrame(records)
        if not equity_curve.empty:
            equity_curve.set_index('date', inplace=True)

        return equity_curve


if __name__ == "__main__":
    print("BacktestEngine tested via jobs/backtest_job.py")
    print("backtest/engine.py: OK")
