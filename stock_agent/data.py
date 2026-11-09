import yfinance as yf
import pandas as pd

def get_stock_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Download daily OHLCV data from Yahoo Finance via yfinance."""
    data = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if data is None or data.empty:
        return pd.DataFrame()

    # yfinance may return MultiIndex columns for some versions/configurations.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.columns = [str(c).title() for c in data.columns]
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Downloaded data is missing columns: {', '.join(missing)}")

    data = data[required].copy()
    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["Open", "High", "Low", "Close"])
    data = data[data["Close"] > 0]
    return data
