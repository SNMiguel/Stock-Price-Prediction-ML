"""
Streamlit live trading dashboard.

Sections:
  1. Portfolio Value  — live from Alpaca
  2. Equity Curve     — historical from DB
  3. Recent Trades    — last 20 from DB
  4. Today's Signals  — latest prediction per ticker
  5. Model Registry   — version history

Run locally:
    streamlit run monitoring/dashboard.py

Deploy:
    Push to GitHub → connect repo on streamlit.io/cloud
    Add secrets in Streamlit Cloud dashboard (same keys as .env)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from datetime import date, timedelta

import config
from data.database import Database
from data.alpaca_feed import AlpacaFeed
from models.registry import ModelRegistry

sns.set_style("whitegrid")

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Trading Dashboard",
    page_icon="📈",
    layout="wide",
)

# ------------------------------------------------------------------
# Cached connections
# ------------------------------------------------------------------

@st.cache_resource
def get_feed():
    return AlpacaFeed(config.ALPACA_API_KEY,
                      config.ALPACA_SECRET_KEY,
                      config.ALPACA_BASE_URL)


@st.cache_resource
def get_db():
    db = Database(config.DB_URL)
    db.create_tables()
    return db


@st.cache_data(ttl=60)
def fetch_account():
    return get_feed().get_account()


@st.cache_data(ttl=300)
def fetch_portfolio_history():
    return get_db().get_portfolio_history()


@st.cache_data(ttl=60)
def fetch_trade_log():
    return get_db().get_trade_log()


@st.cache_data(ttl=60)
def fetch_latest_predictions():
    """Return latest prediction row per ticker from the predictions table."""
    db = get_db()
    rows = []
    end   = date.today().isoformat()
    start = (date.today() - timedelta(days=7)).isoformat()
    for ticker in config.WATCHLIST:
        sentiment = db.get_sentiment(ticker, start, end)
        score = float(sentiment['score'].iloc[-1]) if not sentiment.empty else 0.0
        rows.append({'ticker': ticker, 'latest_sentiment': round(score, 4)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title("📈 Paper Trading Dashboard")
st.caption(f"Watchlist: {', '.join(config.WATCHLIST)}  |  "
           f"Live trading: {'🔴 LIVE' if config.LIVE_TRADING else '🟡 PAPER'}")

if st.button("🔄 Refresh"):
    st.cache_data.clear()

st.divider()

# ------------------------------------------------------------------
# Section 1 — Portfolio Value
# ------------------------------------------------------------------
st.subheader("1. Portfolio Value")

try:
    account = fetch_account()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Equity",    f"${account['equity']:,.2f}")
    col2.metric("Cash Available",  f"${account['cash']:,.2f}")
    col3.metric("Portfolio Value", f"${account['portfolio_value']:,.2f}")
except Exception as e:
    st.error(f"Could not fetch account: {e}")

st.divider()

# ------------------------------------------------------------------
# Section 2 — Equity Curve
# ------------------------------------------------------------------
st.subheader("2. Equity Curve")

history = fetch_portfolio_history()

if history.empty:
    st.info("No portfolio snapshots yet. Run daily_job.py to start recording.")
else:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(history.index, history['value'],
            color='#2E86AB', linewidth=2)
    ax.axhline(y=100_000, color='grey', linestyle='--',
               linewidth=1, alpha=0.6, label='Initial Capital')
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.set_title("Portfolio Equity Over Time", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Drawdown
    values   = history['value'].values
    peak     = np.maximum.accumulate(values)
    drawdown = np.where(peak > 0, (peak - values) / peak * 100, 0)
    max_dd   = float(np.max(drawdown))
    total_r  = (values[-1] - 100_000) / 100_000 * 100

    c1, c2 = st.columns(2)
    c1.metric("Total Return",  f"{total_r:+.2f}%")
    c2.metric("Max Drawdown",  f"{max_dd:.2f}%")

st.divider()

# ------------------------------------------------------------------
# Section 3 — Recent Trades
# ------------------------------------------------------------------
st.subheader("3. Recent Trades")

trades = fetch_trade_log()

if trades.empty:
    st.info("No trades recorded yet.")
else:
    st.dataframe(
        trades.head(20).style.applymap(
            lambda v: 'color: green' if v == 'BUY'
                      else ('color: red' if v == 'SELL' else ''),
            subset=['side']
        ),
        use_container_width=True,
    )

st.divider()

# ------------------------------------------------------------------
# Section 4 — Latest Sentiment (proxy for signal readiness)
# ------------------------------------------------------------------
st.subheader("4. Latest Sentiment Scores")

try:
    preds = fetch_latest_predictions()
    if preds.empty:
        st.info("No sentiment data yet.")
    else:
        for _, row in preds.iterrows():
            score = row['latest_sentiment']
            color = "🟢" if score > 0.05 else ("🔴" if score < -0.05 else "⚪")
            st.write(f"{color} **{row['ticker']}** — sentiment score: `{score}`")
except Exception as e:
    st.warning(f"Could not load sentiment data: {e}")

st.divider()

# ------------------------------------------------------------------
# Section 5 — Model Registry
# ------------------------------------------------------------------
st.subheader("5. Model Registry")

try:
    registry = ModelRegistry()
    versions = registry.list_versions()

    if not versions:
        st.info("No models saved yet. Run train_job.py first.")
    else:
        rows = []
        for v in versions:
            m = v.get('metrics', {})
            rows.append({
                'Version':   v['version_id'],
                'Name':      v['name'],
                'RMSE':      round(m.get('rmse', 0), 4),
                'R²':        round(m.get('r2', 0), 4),
                'Saved':     v['timestamp'][:19].replace('T', ' '),
                'Framework': v['framework'],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
except Exception as e:
    st.warning(f"Could not load registry: {e}")
