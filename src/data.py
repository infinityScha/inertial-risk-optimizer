import yfinance as yf
import pandas as pd
import numpy as np

def get_blue_chip_returns(n_assets=50, start_date=None, end_date=None, years=None):
    """
    Fetches historical daily log returns for top S&P 500 components.
    Priority: start_date takes precedence over years.
    
    Args:
        n_assets (int): Number of assets to return.
        start_date (str/Timestamp): Optional start date.
        end_date (str/Timestamp): Optional end date. Defaults to today.
        years (int): Number of years to look back if start_date is None.
        
    Returns:
        List: List of ticker symbols.
        np.ndarray: Cleaned return matrix (T x N).
    """
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "BRK-B", "JPM", "V", "JNJ", "WMT", "PG",
        "MA", "XOM", "CVX", "HD", "KO", "PEP", "BAC", "PFE", "ABBV", "COST",
        "AVGO", "CSCO", "MCD", "TMO", "CRM", "ADBE", "WFC", "DIS", "PM", "AMD",
        "ORCL", "CMCSA", "NFLX", "INTC", "UPS", "TXN", "HON", "LOW", "MS", "BA",
        "CAT", "IBM", "DE", "GE", "MMM", "GS", "SPGI", "BLK", "NOW", "RTX"
    ][:n_assets]

    # 1. Handle end_date
    if end_date is None:
        end_dt = pd.Timestamp.now()
    else:
        end_dt = pd.to_datetime(end_date)

    # 2. Handle start_date / years logic
    if start_date is not None:
        start_dt = pd.to_datetime(start_date)
    elif years is not None:
        start_dt = end_dt - pd.DateOffset(years=years)
    else:
        # Fallback default
        start_dt = end_dt - pd.DateOffset(years=30)

    print(f"Fetching data from {start_dt.date()} to {end_dt.date()}...")

    # auto_adjust=True handles splits/dividends
    data = yf.download(tickers, start=start_dt, end=end_dt, auto_adjust=True)
    
    if 'Close' in data.columns:
        df = data['Close']
    else:
        df = data

    # Clean data
    # Drop assets missing > 5% of data to ensure we keep those active during the GFC
    df = df.dropna(axis=1, thresh=int(len(df) * 0.95)) 
    df = df.ffill().dropna()

    returns = np.log(df / df.shift(1)).dropna()
    
    actual_tickers = returns.columns.tolist()
    print(f"Final asset count: {len(actual_tickers)}")
    
    return actual_tickers, returns.values