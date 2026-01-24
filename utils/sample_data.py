"""Generate sample stock data for testing when API is unavailable."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_aapl_data(start_date="2020-01-01", end_date="2024-01-01"):
    """
    Generate realistic sample AAPL stock data.
    Based on AAPL's actual price range and patterns from 2020-2024.
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Create date range (business days only)
    dates = pd.bdate_range(start=start, end=end)
    n_days = len(dates)
    
    # Generate realistic price data
    # AAPL was ~$75 in early 2020, ~$185 in late 2023
    np.random.seed(42)  # For reproducibility
    
    # Create upward trend with volatility
    trend = np.linspace(75, 185, n_days)
    volatility = np.random.normal(0, 3, n_days)
    seasonal = 5 * np.sin(np.linspace(0, 8*np.pi, n_days))  # Seasonal patterns
    
    close_prices = trend + volatility + seasonal
    
    # Generate OHLC data
    daily_range = np.random.uniform(0.5, 3, n_days)
    high_prices = close_prices + daily_range
    low_prices = close_prices - daily_range
    open_prices = close_prices + np.random.uniform(-1, 1, n_days)
    
    # Generate volume (AAPL typically 50-150M daily volume)
    volume = np.random.uniform(50_000_000, 150_000_000, n_days)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volume.astype(int)
    }, index=dates)
    
    return data

if __name__ == "__main__":
    data = generate_sample_aapl_data()
    print(data.head())
    print(f"\nShape: {data.shape}")
    print(f"Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")