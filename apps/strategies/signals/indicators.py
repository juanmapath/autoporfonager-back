import numpy as np
import pandas as pd


def MACD(serie: pd.Series, fast_period: int = 28, slow_period: int = 56, signal_period: int = 14):
    ema_fast = serie.ewm(span=fast_period, adjust=False).mean()
    ema_slow = serie.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    return [macd_line, signal_line, macd_histogram]


def ATR(data: pd.DataFrame, window: int) -> pd.Series:
    serie = data.copy()
    serie["High-Low"] = serie["High"] - serie["Low"]
    serie["High-PrevClose"] = (serie["High"] - serie["Close"].shift(1)).abs()
    serie["Low-PrevClose"] = (serie["Low"] - serie["Close"].shift(1)).abs()
    serie["TR"] = serie[["High-Low", "High-PrevClose", "Low-PrevClose"]].max(axis=1)
    return serie["TR"].rolling(window=window).mean()


def RSI(data: pd.DataFrame, window: int = 14) -> pd.Series:
    close = data["Close"]
    change = close.diff()
    change_up = np.where(change > 0, change, 0.0)
    change_down = np.where(change < 0, np.abs(change), 0.0)
    avg_up = pd.Series(change_up, index=close.index).rolling(window).mean()
    avg_down = pd.Series(change_down, index=close.index).rolling(window).mean()
    return ((100.0 * avg_up) / (avg_up + avg_down)).round(2)


def supertrend(data: pd.DataFrame, lookback: int = 28, multiplier: float = 2.0) -> pd.DataFrame:
    df = data.copy().reset_index(drop=True)
    atr = ATR(df, window=lookback)
    hl_avg = (df["High"] + df["Low"]) / 2.0
    upper_band = hl_avg + multiplier * atr
    lower_band = hl_avg - multiplier * atr

    n = len(df)
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    supertrend_vals = np.zeros(n)

    valid_indices = lower_band.dropna().index
    if len(valid_indices) == 0:
        df["supertrend"] = np.nan
        df["st_up"] = np.nan
        df["st_d"] = np.nan
        return df[["supertrend", "st_up", "st_d"]]

    start_idx = valid_indices[0]

    for i in range(start_idx, n):
        if i == start_idx:
            final_upper[i] = upper_band.iloc[i]
            final_lower[i] = lower_band.iloc[i]
        else:
            if (upper_band.iloc[i] < final_upper[i - 1]) or (df.loc[i - 1, "Close"] > final_upper[i - 1]):
                final_upper[i] = upper_band.iloc[i]
            else:
                final_upper[i] = final_upper[i - 1]

            if (lower_band.iloc[i] > final_lower[i - 1]) or (df.loc[i - 1, "Close"] < final_lower[i - 1]):
                final_lower[i] = lower_band.iloc[i]
            else:
                final_lower[i] = final_lower[i - 1]

    for i in range(start_idx, n):
        if i == start_idx:
            supertrend_vals[i] = final_lower[i] if df.loc[i, "Close"] > df.loc[i - 1, "Close"] else final_upper[i]
        else:
            prev_st = supertrend_vals[i - 1]
            close = df.loc[i, "Close"]
            if prev_st == final_upper[i - 1] and close < final_upper[i - 1]:
                supertrend_vals[i] = final_upper[i]
            elif prev_st == final_upper[i - 1] and close > final_upper[i - 1]:
                supertrend_vals[i] = final_lower[i]
            elif prev_st == final_lower[i - 1] and close > final_lower[i - 1]:
                supertrend_vals[i] = final_lower[i]
            elif prev_st == final_lower[i - 1] and close < final_lower[i - 1]:
                supertrend_vals[i] = final_upper[i]
            else:
                supertrend_vals[i] = final_upper[i]

    df["supertrend"] = supertrend_vals
    df["st_up"] = np.where(df["supertrend"] < df["Close"], df["supertrend"], np.nan)
    df["st_d"] = np.where(df["supertrend"] > df["Close"], df["supertrend"], np.nan)
    return df[["supertrend", "st_up", "st_d"]]


def dema(data: pd.DataFrame, window: int, serie: str = "Close") -> pd.Series:
    close = data[serie]
    ema = close.ewm(span=window).mean()
    dema_ema = ema.ewm(span=window).mean()
    return 2 * ema - dema_ema


def slope(data: pd.DataFrame, window: int, serie: str = "Close") -> pd.Series:
    y_vals = data[serie].values
    x = np.arange(window)
    sum_x, sum_x2 = np.sum(x), np.sum(x**2)
    divisor = window * sum_x2 - sum_x**2
    slopes = [np.nan] * (window - 1)
    for i in range(window, len(y_vals) + 1):
        y_slice = y_vals[i - window:i]
        m = (window * np.sum(x * y_slice) - sum_x * np.sum(y_slice)) / divisor
        slopes.append(m)
    return pd.Series(slopes, index=data.index)


def zscore(data: pd.DataFrame, window: int, serie: str = "Close") -> pd.Series:
    series = data[serie]
    ma = series.rolling(window=window).mean()
    stdev = series.rolling(window=window).std()
    return (series - ma) / stdev


def rolling_slope(series: pd.Series, window: int = 10) -> pd.Series:
    y = series.values
    x = np.arange(window)
    sum_x, sum_x2 = np.sum(x), np.sum(x**2)
    divisor = window * sum_x2 - sum_x**2
    slopes = [np.nan] * (window - 1)
    for i in range(window, len(y) + 1):
        y_slice = y[i - window:i]
        m = (window * np.sum(x * y_slice) - sum_x * np.sum(y_slice)) / divisor
        slopes.append(m)
    return pd.Series(slopes, index=series.index)


def get_zscore(series: pd.Series, window: int = 60) -> pd.Series:
    r = series.rolling(window=window)
    return (series - r.mean()) / r.std(ddof=0)
