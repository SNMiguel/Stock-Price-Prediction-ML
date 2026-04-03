# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Set up environment
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt

# Run the full pipeline
python main.py

# Run individual modules for testing
python utils/data_loader.py
python models/neural_network.py
python models/model_comparison.py
```

There is no test suite — each module has an `if __name__ == "__main__":` block that can be run directly to exercise it in isolation.

## Architecture

The pipeline runs sequentially in `main.py`:

1. **Data ingestion** (`utils/data_loader.py` → `StockDataLoader`)  
   Downloads AAPL OHLCV data via `yfinance`. Falls back to `utils/sample_data.py` (synthetic data generator) if the network call fails. After downloading, `add_technical_indicators()` computes 18 derived columns in-place on `self.data`, then `prepare_features()` drops raw OHLCV columns and returns `(X, y, dates)`.

2. **Train/test split** (`main.py`)  
   Chronological 80/20 split — no shuffling, to avoid time-series leakage.

3. **Model training** (`models/model_comparison.py` → `ModelComparison`)  
   - `models/linear_regression.py` (`TraditionalModels`): wraps scikit-learn Linear Regression, Random Forest, and SVR. Each model is stored in `self.models` dict after fitting.  
   - `models/neural_network.py` (`NeuralNetworkModel`): Keras Sequential model. Features are scaled with `StandardScaler` (fit on train, transform on test) before passing to the network. Three architecture variants exist: `'standard'` (default), `'deep'`, `'wide'`.

4. **Evaluation** (`utils/evaluation.py` → `ModelEvaluator`)  
   Calculates MAE, RMSE, R², MAPE. Results are stored inside `ModelComparison.results` keyed by model name. `get_best_model()` selects by lowest RMSE.

5. **Visualization**  
   Plots are written to `results/` (created if absent). `ModelEvaluator` provides `plot_predictions()`, `plot_residuals()`, and `compare_models()`.

## Key design decisions

- The `NeuralNetworkModel` owns its own `StandardScaler`. Scaling happens inside `ModelComparison`, not inside `StockDataLoader`, so traditional models receive raw feature values while the NN receives scaled values.
- `StockDataLoader.prepare_features()` excludes `['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']` from `X` — the target `y` is the raw `Close` price.
- Changing the ticker or date range is done by modifying the `StockDataLoader(...)` call in `main.py`. Sample-data fallback only activates for `ticker="AAPL"`.
