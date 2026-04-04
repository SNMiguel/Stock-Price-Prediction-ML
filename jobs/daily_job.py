"""
Daily trading job — runs once per day after market close (or pre-open).

Execution order:
  1.  Cancel any leftover open orders
  2.  Check market is open today (skip if holiday / weekend)
  3.  Fetch latest price bar for each ticker → cache in DB
  4.  Fetch today's news sentiment for each ticker → cache in DB
  5.  Build feature matrix up to today (leakage-free)
  6.  Load best model from registry per ticker
  7.  Predict next-day close, compute confidence
  8.  Generate signal (BUY / SELL / HOLD)
  9.  Execute signal via OrderManager
  10. Snapshot portfolio value in DB
  11. Send daily summary to Discord

Usage:
    python -m jobs.daily_job
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date
import traceback

import config
from data.database import Database
from data.alpaca_feed import AlpacaFeed
from data.news_sentiment import NewsSentiment
from features.walk_forward import get_features_at
from features.sentiment_features import merge_sentiment
from models.registry import ModelRegistry
from models.ensemble import EnsembleModel
from signals.generator import SignalGenerator
from risk.position_sizer import PositionSizer
from risk.portfolio import Portfolio
from execution.alpaca_broker import AlpacaBroker
from execution.order_manager import OrderManager
from monitoring.alerts import DiscordAlerter


def run() -> None:
    today = date.today().isoformat()
    print(f"\n{'='*55}")
    print(f"  Daily Job — {today}")
    print(f"  Live trading: {config.LIVE_TRADING}")
    print(f"{'='*55}")

    alerter = DiscordAlerter(config.DISCORD_WEBHOOK_URL)

    try:
        # ------------------------------------------------------------------
        # 1. Connections
        # ------------------------------------------------------------------
        db      = Database(config.DB_URL)
        db.create_tables()
        feed    = AlpacaFeed(config.ALPACA_API_KEY,
                             config.ALPACA_SECRET_KEY,
                             config.ALPACA_BASE_URL)
        broker  = AlpacaBroker(config.ALPACA_API_KEY,
                               config.ALPACA_SECRET_KEY,
                               config.ALPACA_BASE_URL)
        sizer   = PositionSizer(max_position_pct=config.MAX_POSITION_PCT)
        portfolio = Portfolio(feed, db)
        order_mgr = OrderManager(broker, portfolio, sizer, db, alerter)
        signal_gen = SignalGenerator(
            threshold=config.SIGNAL_THRESHOLD,
            confidence_threshold=config.CONFIDENCE_THRESHOLD,
        )
        registry = ModelRegistry()
        sentiment_client = NewsSentiment(config.NEWS_API_KEY)

        # ------------------------------------------------------------------
        # 2. Cancel stale orders
        # ------------------------------------------------------------------
        broker.cancel_all_orders()
        print("✓ Cancelled any open orders")

        # ------------------------------------------------------------------
        # 3. Market-open check (skip weekends / holidays)
        # ------------------------------------------------------------------
        if not broker.is_market_open():
            print("Market is closed today — exiting.")
            return

        # ------------------------------------------------------------------
        # 4–9. Per-ticker loop
        # ------------------------------------------------------------------
        all_signals: dict = {}

        for ticker in config.WATCHLIST:
            print(f"\n--- {ticker} ---")

            # 4a. Fetch price history (cached in DB)
            start_date = "2022-01-01"
            df = feed.get_historical_bars(ticker, start_date, today, db=db)
            if df.empty:
                print(f"  ⚠ No price data — skipping {ticker}")
                continue

            # 4b. Fetch sentiment
            sentiment_score = sentiment_client.get_daily_score(ticker, today, db)
            sentiment_df    = db.get_sentiment(ticker, start_date, today)
            print(f"  Sentiment today: {sentiment_score:+.4f}")

            # 5. Build leakage-free features at today's cutoff
            X, y, dates = get_features_at(df, today)
            if len(X) < 2:
                print(f"  ⚠ Not enough feature rows — skipping {ticker}")
                continue

            # Merge sentiment into feature matrix
            import pandas as pd
            import numpy as np
            feature_df   = pd.DataFrame(X, index=dates)
            feature_df   = merge_sentiment(feature_df, sentiment_df)
            X_full       = feature_df.values

            # 6. Load best model
            model, meta = registry.load_best('rmse',
                                             name_prefix=f'ensemble_{ticker}')
            if model is None:
                print(f"  ⚠ No model for {ticker} — run train_job.py first")
                continue
            print(f"  Model RMSE: ${meta['metrics']['rmse']:.4f}")

            # 7. Predict on latest row
            X_today     = X_full[-1].reshape(1, -1)
            predicted   = float(model.predict(X_today)[0])
            current     = float(df['Close'].iloc[-1])

            confidence  = 0.5   # default
            if hasattr(model, 'get_confidence'):
                confidence = float(model.get_confidence(X_today))

            print(f"  Current: ${current:.2f}  Predicted: ${predicted:.2f}  "
                  f"Conf: {confidence:.2f}")

            # 8. Generate signal
            signal = signal_gen.generate(current, predicted, confidence)
            all_signals[ticker] = signal
            print(f"  Signal: {signal['signal']}  Δ{signal['delta_pct']:+.2f}%")

            # 9. Execute
            order = order_mgr.execute_signal(signal, ticker, df)
            if order:
                print(f"  ✓ Order submitted: {order.get('id', order)}")
            else:
                print(f"  – No order placed (HOLD or risk limit)")

        # ------------------------------------------------------------------
        # 10. Snapshot portfolio
        # ------------------------------------------------------------------
        portfolio_value = portfolio.get_portfolio_value()
        portfolio.snapshot(today)
        print(f"\nPortfolio value: ${portfolio_value:,.2f}")

        # ------------------------------------------------------------------
        # 11. Discord summary
        # ------------------------------------------------------------------
        alerter.send_daily_summary(all_signals, portfolio_value)
        print("✓ Discord summary sent")
        print(f"\n{'='*55}")
        print("  Daily job complete")
        print(f"{'='*55}\n")

    except Exception as exc:
        msg = f"daily_job failed: {exc}\n{traceback.format_exc()}"
        print(f"\n🚨 {msg}")
        alerter.send_error(msg)
        raise


if __name__ == "__main__":
    run()
