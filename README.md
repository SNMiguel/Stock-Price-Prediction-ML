# QuantPilot — Automated Paper Trading System

An end-to-end automated trading system that combines ensemble ML models, live market data, news sentiment analysis, and risk management to trade AAPL, MSFT, and GOOGL on Alpaca's paper trading platform.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.0-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 What It Does

Every weekday after market close, QuantPilot automatically:
1. Fetches the latest price bar and news sentiment for each ticker
2. Runs leakage-free feature engineering (indicators computed only on past data)
3. Loads the best trained ensemble model per ticker from the registry
4. Generates a BUY / SELL / HOLD signal with confidence score
5. Sizes the position using ATR-based risk management (max 15% portfolio exposure)
6. Submits the order to Alpaca and logs it to the database
7. Snapshots portfolio value and sends a Discord summary

Every Sunday, models are retrained on fresh data and saved to the registry only if RMSE improves.

**Live dashboard:** [quantpilot.streamlit.app](https://quantpilot.streamlit.app)

## 🏗️ Architecture

```
QuantPilot/
├── config.py                     # All settings loaded from .env
│
├── data/
│   ├── database.py               # PostgreSQL via SQLAlchemy (Neon.tech)
│   ├── alpaca_feed.py            # Live + historical price data (IEX feed)
│   └── news_sentiment.py         # NewsAPI + VADER sentiment scoring
│
├── features/
│   ├── indicators.py             # 18 technical indicators (MA, RSI, MACD, BB...)
│   ├── walk_forward.py           # Leakage-free feature slicing by cutoff date
│   └── sentiment_features.py     # Merges sentiment scores onto feature matrix
│
├── models/
│   ├── ensemble.py               # Ridge meta-learner over OOF base predictions
│   ├── registry.py               # JSON manifest + joblib/keras persistence
│   ├── linear_regression.py      # LR, Random Forest, SVR (scikit-learn)
│   ├── neural_network.py         # LSTM + standard/deep/wide architectures
│   └── model_comparison.py       # Trains and evaluates all models side-by-side
│
├── training/
│   ├── walk_forward_trainer.py   # Expanding-window cross-validation + retrain
│   └── metrics.py                # Sharpe, max drawdown, Calmar, win rate, profit factor
│
├── signals/
│   └── generator.py              # BUY/SELL/HOLD from predicted vs current price
│
├── risk/
│   ├── position_sizer.py         # ATR-based sizing with portfolio exposure cap
│   └── portfolio.py              # Syncs positions with Alpaca, tracks drawdown
│
├── execution/
│   ├── alpaca_broker.py          # Order submission, position queries
│   └── order_manager.py          # Full signal → order pipeline with risk checks
│
├── backtest/
│   ├── engine.py                 # Event-driven backtester (next-day open fill)
│   └── report.py                 # Financial metrics + equity curve chart
│
├── monitoring/
│   ├── dashboard.py              # Streamlit live dashboard (5 sections)
│   └── alerts.py                 # Discord webhook notifications
│
├── jobs/
│   ├── daily_job.py              # Runs the full trade pipeline for all tickers
│   ├── train_job.py              # Retrains models, saves if RMSE improved
│   └── backtest_job.py           # On-demand historical strategy evaluation
│
└── .github/workflows/
    ├── daily_trade.yml           # Cron: Mon–Fri 21:30 UTC (5:30 PM ET)
    └── weekly_retrain.yml        # Cron: Sunday 02:00 UTC
```

## 📊 Models

The system trains one **ensemble model per ticker** using a stacked architecture:

- **Base models**: Linear Regression, Random Forest, SVR (scikit-learn) + Neural Network (Keras)
- **Meta-learner**: Ridge regression trained on out-of-fold predictions
- **Validation**: 5-fold expanding-window walk-forward (no data leakage)
- **Registry**: Models are versioned in JSON — production model only updates if RMSE improves

## 🖥️ Dashboard

Five sections updated live:

| Section | Source |
|---|---|
| Portfolio Value | Alpaca account API |
| Equity Curve + Drawdown | `portfolio_snapshots` DB table |
| Recent Trades | `trades` DB table |
| Sentiment Scores | `sentiment` DB table |
| Model Registry | `models/saved/registry.json` |

## 🚀 Local Setup

```bash
git clone https://github.com/SNMiguel/QuantPilot.git
cd QuantPilot

python -m venv venv
source venv/Scripts/activate   # Windows Git Bash

pip install -r requirements.txt

# Copy and fill in your credentials
cp .env.example .env
```

**Required `.env` keys:**

```
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
ALPACA_BASE_URL=https://paper-api.alpaca.markets
DB_URL=                   # Neon.tech PostgreSQL connection string
NEWS_API_KEY=             # newsapi.org
DISCORD_WEBHOOK_URL=      # optional — alerts channel
LIVE_TRADING=false        # set true only when ready for real money
```

**Initialize and test:**

```bash
python -m data.database         # Create tables
python -m data.alpaca_feed      # Verify Alpaca connection
python -m jobs.train_job        # Train all 3 ticker models (~5 min)
python -m jobs.daily_job        # Run one full trade cycle
streamlit run monitoring/dashboard.py
```

**Backtest a ticker:**

```bash
python -m jobs.backtest_job --ticker AAPL --start 2024-01-01
```

## ⚙️ GitHub Actions

Two automated workflows run on schedule — no server required:

| Workflow | Schedule | Job |
|---|---|---|
| `daily_trade.yml` | Mon–Fri 21:30 UTC | `jobs/daily_job.py` |
| `weekly_retrain.yml` | Sunday 02:00 UTC | `jobs/train_job.py` |

**Required GitHub secrets** (Settings → Secrets → Actions):
`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPACA_BASE_URL`, `DB_URL`, `NEWS_API_KEY`, `DISCORD_WEBHOOK_URL`, `LIVE_TRADING`

## 🛡️ Risk Management

- **ATR-based position sizing**: `(portfolio × risk_per_trade) / (ATR × multiplier)`
- **Hard exposure cap**: single position ≤ 15% of portfolio value
- **Confidence gate**: signals below `CONFIDENCE_THRESHOLD` (0.60) are ignored
- **LIVE_TRADING gate**: must be explicitly set to `true` to submit real orders
- **Cash account**: avoids PDT rule (no margin, no 3-trade-per-week limit)

## 🛠️ Technologies

| Layer | Stack |
|---|---|
| ML | scikit-learn, TensorFlow/Keras |
| Data | yfinance, Alpaca Markets API, NewsAPI, VADER |
| Database | PostgreSQL (Neon.tech) via SQLAlchemy |
| Dashboard | Streamlit |
| Automation | GitHub Actions |
| Alerts | Discord webhooks |

## 👤 Author

**Miguel Shema Ngabonziza**
- LinkedIn: [linkedin.com/in/migztech](https://linkedin.com/in/migztech)
- GitHub: [github.com/SNMiguel](https://github.com/SNMiguel)
- Portfolio: [migztech.vercel.app](https://migztech.vercel.app)

---

⭐ If you found this project useful, consider giving it a star!
