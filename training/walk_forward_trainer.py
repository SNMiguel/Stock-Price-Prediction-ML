"""
Walk-forward training pipeline.

Replaces the single 80/20 split in main.py with a rolling expanding-window
approach that avoids data leakage and gives a more honest estimate of how
the model will perform on unseen future data.

For each of n_splits folds:
  - Train on all data up to cutoff date
  - Evaluate on the next step of unseen data
  - Collect metrics

After all folds, retrain on the most recent retrain_window_days and save
to the model registry if RMSE improved.
"""
import numpy as np
import pandas as pd

from features.walk_forward import get_features_at
from features.sentiment_features import merge_sentiment
from models.model_comparison import ModelComparison
from models.ensemble import EnsembleModel
from models.registry import ModelRegistry
from utils.evaluation import ModelEvaluator


class WalkForwardTrainer:
    """Expanding-window walk-forward trainer for all tickers."""

    def __init__(self, n_splits: int = 5,
                 retrain_window_days: int = 500):
        """
        Args:
            n_splits:             Number of walk-forward validation folds.
            retrain_window_days:  How many most-recent trading days to use
                                  for the final production retrain.
        """
        self.n_splits             = n_splits
        self.retrain_window_days  = retrain_window_days

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame,
              sentiment_df: pd.DataFrame,
              ticker: str = None) -> dict:
        """
        Run walk-forward validation then retrain on the most recent window.

        Args:
            df:            Raw OHLCV DataFrame with DatetimeIndex.
            sentiment_df:  Sentiment scores DataFrame (date index, 'score' col).
                           Can be empty — sentiment will default to 0.0.
            ticker:        Ticker symbol e.g. 'AAPL'. Used as registry key prefix.

        Returns:
            Dict of averaged metrics across all folds:
            {'rmse': float, 'mae': float, 'r2': float, 'mape': float}
        """
        if df is None or df.empty:
            raise ValueError(
                f"Empty DataFrame for ticker '{ticker}'. "
                "Check data fetch — Alpaca may have fallen back to yfinance with no result."
            )

        print(f"\n{'='*60}")
        print(f"Walk-Forward Training  —  {ticker or 'unknown'}")
        print(f"Data: {len(df)} rows  |  Folds: {self.n_splits}  |  Final window: {self.retrain_window_days} days")
        print('='*60)

        fold_metrics = self._run_folds(df, sentiment_df)

        if not fold_metrics:
            print("⚠ No valid folds completed. Check data length.")
            return {}

        avg = self._average_metrics(fold_metrics)

        print(f"\n{'='*60}")
        print("Walk-Forward Averaged Metrics")
        print('='*60)
        print(f"  RMSE : ${avg['rmse']:.4f}")
        print(f"  MAE  : ${avg['mae']:.4f}")
        print(f"  R²   : {avg['r2']:.4f}")
        print(f"  MAPE : {avg['mape']:.2f}%")

        # Final production retrain
        self._final_retrain(df, sentiment_df, ticker, avg)

        return avg

    # ------------------------------------------------------------------
    # Folds
    # ------------------------------------------------------------------

    def _run_folds(self, df: pd.DataFrame,
                   sentiment_df: pd.DataFrame) -> list:
        """Run n_splits expanding-window folds and return per-fold metrics."""
        dates     = df.index
        n         = len(dates)
        step      = n // (self.n_splits + 1)
        fold_results = []

        for i in range(1, self.n_splits + 1):
            train_end_idx = i * step
            test_end_idx  = min((i + 1) * step, n - 1)

            cutoff_train = dates[train_end_idx].strftime('%Y-%m-%d')
            cutoff_test  = dates[test_end_idx].strftime('%Y-%m-%d')

            print(f"\n--- Fold {i}/{self.n_splits} "
                  f"| train → {cutoff_train}  test → {cutoff_test} ---")

            # Features up to test end (no leakage)
            X_all, y_all, dates_all = get_features_at(df, cutoff_test)

            # Merge sentiment
            X_all = self._merge_sentiment_array(
                X_all, dates_all, sentiment_df
            )

            # Split
            train_mask = dates_all <= cutoff_train
            test_mask  = ~train_mask

            X_train, y_train = X_all[train_mask], y_all[train_mask]
            X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

            if len(X_train) < 60 or len(X_test) < 10:
                print(f"  ⚠ Skipping fold {i} — not enough data "
                      f"(train={len(X_train)}, test={len(X_test)})")
                continue

            # Train all models
            comparison = ModelComparison()
            comparison.train_all_models(
                X_train, y_train, input_dim=X_train.shape[1]
            )

            # Evaluate on test set
            comparison.evaluate_all_models(
                X_test, y_test, dates_all[test_mask]
            )

            best_name, best_result = comparison.get_best_model()
            m = best_result['metrics']
            fold_results.append({
                'rmse': m['RMSE'],
                'mae':  m['MAE'],
                'r2':   m['R²'],
                'mape': m['MAPE'],
            })

            print(f"  Best: {best_name}  RMSE=${m['RMSE']:.2f}  R²={m['R²']:.4f}")

        return fold_results

    # ------------------------------------------------------------------
    # Final retrain
    # ------------------------------------------------------------------

    def _final_retrain(self, df: pd.DataFrame,
                       sentiment_df: pd.DataFrame,
                       ticker: str,
                       fold_avg: dict) -> None:
        """
        Retrain on the most recent retrain_window_days.
        Save to registry only if RMSE beats the current best.
        """
        print(f"\n{'='*60}")
        print("Final Production Retrain")
        print('='*60)

        recent_df = df.iloc[-self.retrain_window_days:]
        cutoff    = recent_df.index[-1].strftime('%Y-%m-%d')

        X, y, dates = get_features_at(recent_df, cutoff)
        X = self._merge_sentiment_array(X, dates, sentiment_df)

        split    = int(0.8 * len(X))
        X_train  = X[:split];   y_train = y[:split]
        X_test   = X[split:];   y_test  = y[split:]

        # Train all base models
        comparison = ModelComparison()
        comparison.train_all_models(
            X_train, y_train, input_dim=X_train.shape[1]
        )

        # Train ensemble on top of traditional models
        ensemble = EnsembleModel(
            dict(comparison.traditional_models.models)
        )
        ensemble.fit(X_train, y_train, n_splits=3)

        # Evaluate ensemble
        evaluator = ModelEvaluator()
        preds     = ensemble.predict(X_test)
        metrics   = evaluator.calculate_metrics(y_test, preds, 'Ensemble')

        new_rmse = metrics['RMSE']
        reg_key  = f'ensemble_{ticker}' if ticker else 'ensemble'

        # Compare against current registry best
        registry                 = ModelRegistry()
        _, current_meta          = registry.load_best('rmse',
                                                       name_prefix=reg_key)
        current_rmse             = (current_meta['metrics'].get('rmse', float('inf'))
                                    if current_meta else float('inf'))

        print(f"  New RMSE    : ${new_rmse:.4f}")
        print(f"  Current best: ${current_rmse:.4f}")

        if new_rmse < current_rmse:
            registry.save(
                ensemble,
                name=reg_key,
                metrics={
                    'rmse': new_rmse,
                    'mae':  metrics['MAE'],
                    'r2':   metrics['R²'],
                    'mape': metrics['MAPE'],
                },
                framework='sklearn',
            )
        else:
            print(f"  Registry unchanged — new model did not improve.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _merge_sentiment_array(self, X: np.ndarray,
                                dates: pd.DatetimeIndex,
                                sentiment_df: pd.DataFrame) -> np.ndarray:
        """
        Attach sentiment scores as an extra column on X.
        Converts to/from DataFrame to use merge_sentiment().
        """
        n_features = X.shape[1]
        col_names  = [f'f{i}' for i in range(n_features)]

        feature_df = pd.DataFrame(X, index=dates, columns=col_names)
        merged     = merge_sentiment(feature_df, sentiment_df)

        return merged.values

    @staticmethod
    def _average_metrics(fold_metrics: list) -> dict:
        keys = fold_metrics[0].keys()
        return {
            k: float(np.mean([m[k] for m in fold_metrics]))
            for k in keys
        }


if __name__ == "__main__":
    import config
    from data.database import Database
    from data.alpaca_feed import AlpacaFeed
    from datetime import date, timedelta

    db   = Database(config.DB_URL)
    feed = AlpacaFeed(config.ALPACA_API_KEY,
                      config.ALPACA_SECRET_KEY,
                      config.ALPACA_BASE_URL)

    ticker     = "AAPL"
    end        = date.today().strftime('%Y-%m-%d')
    start      = (date.today() - timedelta(days=config.TRAIN_LOOKBACK_DAYS)
                  ).strftime('%Y-%m-%d')

    print(f"Fetching {ticker} data {start} → {end} ...")
    df           = feed.get_historical_bars(ticker, start, end, db=db)
    sentiment_df = db.get_sentiment(ticker, start, end)

    trainer = WalkForwardTrainer(n_splits=3)
    metrics = trainer.train(df, sentiment_df, ticker=ticker)

    print(f"\nFinal averaged metrics: {metrics}")
    print("training/walk_forward_trainer.py: OK")
