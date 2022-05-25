from vqti.load import load_eod
from vqti.profile import time_this
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List

def pure_python_sma(close: List[float], m: int=10) -> List[float]:
	assert m >= 1, 'Window must be positive.'
	result = [None] * (m-1)

	accum = sum(close[:m])
	result.append(accum / m)

	for i in range(m, len(close)):
		accum += close[i]
		accum -= close[i-m]
		result.append(accum / m)

	return result


def pure_python_rolling_std_dev(close: List[float], m: int=40) -> List[float]:

	sma = pure_python_sma(close, m)

	numerator_cusum = [None] * m
	accum = 0
	for i in range(m, len(close)):
		accum += (close[i] - sma[i])**2
		numerator_cusum.append(accum)

	numerator_rolling_cusum = [None] * (2*m)
	for i in range(2*m, len(close)):
		value = numerator_cusum[i] - numerator_cusum[i-m]
		numerator_rolling_cusum.append(value)

	rolling_std = [None] * (2*m)
	for i in range(2*m, len(close)):
		value = (numerator_rolling_cusum[i] / (m - 1)) ** 0.5
		rolling_std.append(value)

	return rolling_std


def pure_python_bollinger_bands(df: pd.Series, m: int = 40) -> pd.Series:

    close = df.close.tolist()
    
    sma = pure_python_sma(close, m)
    rolling_std_dev = pure_python_rolling_std_dev(close, m)
    
    df = df.assign(sma=pd.Series(sma, index=df.index))
    df = df.assign(
        upper_band=df.sma + 2 * pd.Series(rolling_std_dev, index=df.index),
        lower_band=df.sma - 2 * pd.Series(rolling_std_dev, index=df.index)
    )
    
    return df


def BB_signal(df: pd.Series, m: int = 40) -> pd.Series:
    
    BB = pure_python_bollinger_bands(df, m)
    
    sell = df['close'] > BB['upper_band']
    sell_sma = df['close'] < BB['sma']
    buy = df['close'] < BB['lower_band']
    buy_sma = df['close'] > BB['sma']
    
    
    return (1*(buy+buy_sma) - 1*(sell+sell_sma))
    

if __name__ == '__main__':
    
    signal = BB_signal(df = load_eod('AWU'), m = 20)
    print(signal.value_counts())

