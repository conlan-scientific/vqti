import pandas as pd

def calculate_simple_moving_average(series: pd.Series, n: int=20) -> pd.Series:
    """Calculates the simple moving average"""
    return series.rolling(n).mean()

def calculate_simple_moving_sample_stdev(series: pd.Series, n: int=20) -> pd.Series:
    """Calculates the simple moving average"""
    return series.rolling(n).std()

def calculate_bollinger_bands(series: pd.Series, n: int=20) -> pd.DataFrame:
    """
    Calculates the Bollinger Bands and returns them as a dataframe
    """
    sma = calculate_simple_moving_average(series, n)
    stdev = calculate_simple_moving_sample_stdev(series, n)

    return pd.DataFrame({
        'middle': sma,
        'upper': sma + 2 * stdev,
        'lower': sma - 2 * stdev
    })

def calculate_percent_b(series: pd.Series, n: int=20) -> pd.DataFrame:
    df = calculate_bollinger_bands(series)
    return 100 * (series - df.lower) / (df.upper - df.lower)

