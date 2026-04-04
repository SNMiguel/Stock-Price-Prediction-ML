"""
Order lifecycle orchestration.
Coordinates risk checks, position sizing, order submission, logging, and alerts.
This is the single entry point for any trade decision in the system.
"""


class OrderManager:
    """
    Orchestrates the full order lifecycle for a single signal.

    Checks risk limits → sizes position → submits order →
    logs to DB → sends Discord alert.
    """

    def __init__(self, broker, portfolio, sizer, db, alerts):
        """
        Args:
            broker:    AlpacaBroker instance.
            portfolio: Portfolio instance.
            sizer:     PositionSizer instance.
            db:        Database instance.
            alerts:    DiscordAlerter instance.
        """
        self.broker    = broker
        self.portfolio = portfolio
        self.sizer     = sizer
        self.db        = db
        self.alerts    = alerts

    def execute_signal(self, signal: dict, ticker: str,
                       price_df, dry_run: bool = False) -> dict | None:
        """
        Execute a trading signal end-to-end.

        Args:
            signal:   Output from SignalGenerator.generate().
            ticker:   Stock symbol e.g. 'AAPL'.
            price_df: OHLCV DataFrame used to compute ATR.
            dry_run:  If True, log and print but do NOT submit to Alpaca.

        Returns:
            Order dict if an order was submitted, None otherwise
            (HOLD signal, risk blocked, or zero shares).
        """
        action = signal.get('signal', 'HOLD')

        # Step 1 — skip holds immediately
        if action == 'HOLD':
            print(f"  {ticker}: HOLD — no order submitted.")
            return None

        current_price = signal['current']

        # Step 2 — calculate ATR and size position
        atr    = self.sizer.calculate_atr(price_df)
        qty    = self.sizer.size(
            self.portfolio.get_portfolio_value(),
            current_price,
            atr,
        )

        if qty <= 0:
            print(f"  {ticker}: position size rounded to 0 — skipping.")
            return None

        # Step 3 — risk limit check
        if not self.portfolio.is_within_limits(ticker, qty, current_price):
            print(f"  {ticker}: blocked by risk limit — skipping.")
            return None

        side = action.lower()   # 'buy' or 'sell'

        print(f"  {ticker}: {action}  {qty} shares @ ~${current_price:.2f}"
              f"  (ATR=${atr:.2f}, confidence={signal['confidence']:.2f})")

        # Step 4 — dry run exits here
        if dry_run:
            print(f"  [DRY RUN] Order NOT submitted.")
            return {
                'id':     'dry-run',
                'ticker': ticker,
                'side':   side,
                'qty':    qty,
                'status': 'dry_run',
            }

        # Step 5 — submit order
        try:
            order = self.broker.submit_order(ticker, qty, side)
        except Exception as e:
            print(f"  ⚠ Order submission failed for {ticker}: {e}")
            self.alerts.send_error(f"Order failed {ticker} {action}: {e}")
            return None

        # Step 6 — log to database
        self.db.log_trade(
            order_id=order['id'],
            ticker=ticker,
            side=action,
            qty=qty,
            price=current_price,
            status=order['status'],
        )

        # Step 7 — send Discord alert
        self.alerts.send_order_alert(ticker, action, qty, current_price)

        print(f"  ✓ Order submitted: {order['id']}  status={order['status']}")
        return order


if __name__ == "__main__":
    print("OrderManager requires live broker/portfolio/db — tested via daily_job.py")
    print("execution/order_manager.py: OK")
