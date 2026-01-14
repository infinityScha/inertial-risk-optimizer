import yfinance as yf
import pandas as pd
import numpy as np

def get_blue_chip_returns(years=15, n_assets=50):
    """
    Fetches historical daily log returns for top S&P 500 components.
    
    Args:
        years (int): Number of years of history.
        n_assets (int): Number of assets to return (sorted by market cap).
        
    Returns:
        List: List of ticker symbols.
        np.ndarray: Cleaned return matrix (T x N).
    """
    # hardcoded list of major players with long histories.
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "BRK-B", "JPM", "V", "JNJ", "WMT", "PG",
        "MA", "XOM", "CVX", "HD", "KO", "PEP", "BAC", "PFE", "ABBV", "COST",
        "AVGO", "CSCO", "MCD", "TMO", "CRM", "ADBE", "WFC", "DIS", "PM", "AMD",
        "ORCL", "CMCSA", "NFLX", "INTC", "UPS", "TXN", "HON", "LOW", "MS", "BA",
        "CAT", "IBM", "DE", "GE", "MMM", "GS", "SPGI", "BLK", "NOW", "RTX"
    ][:n_assets]

    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(years=years)

    # Use auto_adjust=True to get 'Close' which will be the adjusted close
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    
    # Access 'Close' instead of 'Adj Close'
    if 'Close' in data.columns:
        df = data['Close']
    else:
        # Fallback for single-ticker edge cases or different structures
        df = data

    df = df.dropna(axis=1, thresh=int(len(df) * 0.9)) 
    df = df.ffill().dropna()

    returns = np.log(df / df.shift(1)).dropna()
    return returns.columns.tolist(), returns.values


def get_random_returns_from_cov(cov_matrix, years=15):
    """
    Generates synthetic daily log returns from a given covariance matrix.
    
    Args:
        cov_matrix (np.ndarray): Covariance matrix (N x N).
        years (int): Number of years of history.
    Returns:
        np.ndarray: Simulated return matrix (T x N).
    """
    n_assets = cov_matrix.shape[0]
    trading_days_per_year = 252
    total_days = years * trading_days_per_year

    # Generate random returns
    mean_returns = np.zeros(n_assets)
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, size=total_days)

    return returns