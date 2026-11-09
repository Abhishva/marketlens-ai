import pandas as pd

FEATURE_COLUMNS = [
    "Return_1d", "Return_5d", "SMA_5_Ratio", "SMA_20_Ratio",
    "SMA_50_Ratio", "RSI", "MACD", "MACD_Signal",
    "Volatility_10d", "Volume_Ratio"
]

def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create technical indicators using current and past observations only."""
    df = data.copy()
    close = df["Close"]
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, float("nan"))

    df["Return_1d"] = close.pct_change()
    df["Return_5d"] = close.pct_change(5)
    df["SMA_5_Ratio"] = close / close.rolling(5).mean() - 1
    df["SMA_20_Ratio"] = close / close.rolling(20).mean() - 1
    df["SMA_50_Ratio"] = close / close.rolling(50).mean() - 1
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Volatility_10d"] = df["Return_1d"].rolling(10).std()
    volume_avg = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / volume_avg.replace(0, float("nan"))

    # Label whether the next session's close is higher than the current close.
    df["Target"] = (close.shift(-1) > close).astype(int)
    df.loc[df.index[-1], "Target"] = float("nan")
    return df.replace([float("inf"), float("-inf")], float("nan"))
