from setuptools import setup
from vqti.load import EOD_DIR, load_eod
from vqti.profile import time_this
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict, NewType
import os
import glob
from pathlib import Path
from IPython import embed as ipython_embed
import unittest


def numpy_matrix_wma(values: pd.Series, m: int) -> pd.Series:
	assert m >= 1, 'Period must be a positive integer'
	assert type(m) is int, 'Period must be a positive integer'
	assert len(values) >= m, 'Values must be >= period m'
	n = values.shape[0]
	weights = []
	denom = (m * (m + 1)) / 2
	for i in range(1, m + 1):
		x = i / denom
		weights.append(x)

	weights = np.array(weights)
	# Exit early if m greater than length of values
	if m > n:
		return np.array([np.nan] * n)
    
	# Front padding of series
	front_pad = max(m - 1, 0)

	# Initialize the output array
	y = np.empty((n,))

    # Pad with na values
	y[:front_pad] = np.nan

    # Build a matrix to multiply with weight vector
	q = np.empty((n - front_pad, m))
	for j in range(m):
		q[:,j] = values[j:(j+n-m+1)]

	y[front_pad: len(values)] = q.dot(weights)

	return y

def numpy_matrix_hma(values: np.ndarray, m: int=10) -> np.array:
	assert m >= 1, 'Period must be a positive integer'
	assert type(m) is int, 'Period must be a positive integer'
	assert len(values) >= m, 'Values must be >= period m'
	return numpy_matrix_wma((2* numpy_matrix_wma(values, int(m/2))) - (numpy_matrix_wma(values, m)), int(np.sqrt(m)))

def hma_trend_signal(series: pd.Series, m: int=49) -> pd.Series:
    hull_ma = pd.Series(numpy_matrix_hma(series.values, m))
    trend = np.sign(hull_ma - hull_ma.shift(1))
    signal = np.where(trend > trend.shift(1), 1, 0)
    signal = np.where(trend < trend.shift(1), -1, signal)
    return signal

def wma_trend_signal(series: pd.Series, m: int=30) -> pd.Series:
    wma = pd.Series(numpy_matrix_wma(series.values, m))
    trend = np.sign(wma - wma.shift(1))
    signal = np.where(trend > trend.shift(1), 1, 0)
    signal = np.where(trend < trend.shift(1), -1, signal)
    return signal

def rolling_volatility(series, m):
    return series.rolling(m).std() * np.sqrt(252/m)
 
def hma_zscore(series: pd.Series, m1: int=16, m2: int=81) -> pd.Series:
    assert m1 < m2, "m1 must be less than m2"
    hma1 = pd.Series(numpy_matrix_hma(series.values, m1), index=series.index)
    hma2 = pd.Series(numpy_matrix_hma(series.values, m2), index=series.index) 
    vol = rolling_volatility(series, m2)
    return (hma1 - hma2) / vol #volatility should be on the same length as the indicator

def hma_zscore_signal(series: pd.Series, m1: int=16, m2: int=81):
    zscore = hma_zscore(series, m1, m2)
    zscore_sign = np.sign(zscore)
    zscore_shifted_sign = zscore_sign.shift(1, axis=0)
    return zscore_sign * (zscore_sign != zscore_shifted_sign)

def hma_macd_signal(series: pd.Series, m1: int=16, m2: int=49, sig: int=9) -> pd.Series:
    assert m1 < m2, "m1 must be less than m2"
    assert sig < m1, 'signal line must be less than m1'
    hma1 = pd.Series(numpy_matrix_hma(series.values, m1), index=series.index) 
    hma2 = pd.Series(numpy_matrix_hma(series.values, m2), index=series.index)
    macd = hma1 - hma2
    macd_sig = pd.Series(numpy_matrix_hma(macd.values, sig), index=series.index)
    hist = macd - macd_sig
    hist_sign = np.sign(hist)
    hist_shifted_sign = hist_sign.shift(1, axis=0)
    return hist_sign * (hist_sign != hist_shifted_sign)

def hma_price_crossover(series: pd.Series, m: int=16):
    series = np.array(series)
    hull_ma = pd.Series(numpy_matrix_hma(series, m))
    sign = np.where(hull_ma > series, 1, 0)
    sign = pd.Series(np.where(hull_ma < series, -1, sign))
    price_crossover = np.where(sign > sign.shift(1), 1, 0)
    price_crossover = np.where(sign < sign.shift(1), -1, price_crossover)
    return price_crossover 

def hma_crossover(series: pd.Series, m1: int=16, m2: int=81)-> pd.Series:
    fast_hma = pd.Series(numpy_matrix_hma(series, m1))
    slow_hma = pd.Series(numpy_matrix_hma(series, m2))
    sign = np.sign(fast_hma - slow_hma)
    crossover = np.where(sign > sign.shift(1), 1, 0)
    crossover = np.where(sign < sign.shift(1), -1, crossover)
    return crossover

def atr(dataframe: pd.DataFrame, n: int=14,):
    assert pd.Index(['close', 'high', 'low']).isin(df.columns), 'Data frame must have high, low, close in columns'
    high_low = dataframe['high'] - dataframe['low']
    high_close = np.abs(dataframe['high'] - dataframe['close'].shift())
    low_close = np.abs(dataframe['low'] - dataframe['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(n).sum()/n
    return atr

def volatility(series: pd.Series, n: int=14):
    log_returns = pd.Series(np.log(series / series.shift(-1)))
    vol = log_returns.rolling(window=n).std()*np.sqrt(n)
    return vol

def pandas_ema(series, m: int=5):
    return series.ewm(span=m, adjust=False).mean()

def ema_trend_signal(series: pd.Series, m: int=49) -> pd.Series:
    ema = pd.Series(pandas_ema(series.values, m))
    trend = np.sign(ema - ema.shift(1))
    signal = np.where(trend > trend.shift(1), 1, 0)
    signal = np.where(trend < trend.shift(1), -1, signal)
    return signal


"""
Standardization ideas (also signal line ideas)
+ Using trends and signs(DONE)
+ Crossover of two different HMAs
+ Crossover of an HMA and the price
+ Difference between two HMAs divided by volatility (z-score units) (similar to MACD)
"""
class TestSignals(unittest.TestCase, pd.Series):
    def setUp(self, series: pd.Series):
        self.series = series
        return
     
    def test_hma_trend_signal(self):
        ## generate the signals 
        signal = hma_trend_signal(self.series, 49)
        ## find the indices where signals = 1 or -1
        signal_index = signal.loc[signal!=0].index
        truth_case = signal_index
        ## find the trends from the indicator
        hull_ma = numpy_matrix_hma(self.series, 49)
        trend = np.sign(hull_ma - hull_ma.shift(1))
        trend = trend.fillna(0)
        ## find the indices where signals = 1 or -1
        trend_index = trend.loc[trend!=trend.shift(1)].index
        trend_index = trend_index.delete([0, 1])
        ##assert the two indices are equal
        np.testing.assert_array_equal(trend_index, truth_case)
    
    def test_wma_trend_signal(self, series: pd.Series):
        ## generate the signals 
        signal = wma_trend_signal(self.series, 49)
        ## find the indices where signals = 1 or -1
        signal_index = signal.loc[signal!=0].index
        truth_case = signal_index
        ## find the trends from the indicator
        hull_ma = numpy_matrix_wma(self.series, 49)
        trend = np.sign(hull_ma - hull_ma.shift(1))
        trend = trend.fillna(0)
        ## find the indices where signals = 1 or -1
        trend_index = trend.loc[trend!=trend.shift(1)].index
        trend_index = trend_index.delete([0, 1])
        ##assert the two indices are equal
        np.testing.assert_array_equal(trend_index, truth_case)
    
    def test_hma_zscore_signal(self, series: pd.Series):
        ## generate the signals 
        signal = hma_zscore_signal(self.series, 16, 81)
        ## find the indices where signals = 1 or -1
        signal_index = signal.loc[signal!=0].index
        truth_case = signal_index
        ## find the trends from the indicator
        hull_ma = hma_zscore_signal(self.series, 16, 81)
        trend = np.sign(hull_ma - hull_ma.shift(1))
        trend = trend.fillna(0)
        ## find the indices where signals = 1 or -1
        trend_index = trend.loc[trend!=trend.shift(1)].index
        trend_index = trend_index.delete([0, 1])
        ##assert the two indices are equal
        np.testing.assert_array_equal(trend_index, truth_case)
    
    def test_hma_macd_signal(self, series: pd.Series):
        ## generate the signals 
        signal = hma_macd_signal(self.series, 16, 49, 9)
        ## find the indices where signals = 1 or -1
        signal_index = signal.loc[signal!=0].index
        truth_case = signal_index
        ## find the trends from the indicator
        hull_ma = hma_macd_signal(self.series, 16, 49, 9)
        trend = np.sign(hull_ma - hull_ma.shift(1))
        trend = trend.fillna(0)
        ## find the indices where signals = 1 or -1
        trend_index = trend.loc[trend!=trend.shift(1)].index
        trend_index = trend_index.delete([0, 1])
        ##assert the two indices are equal
        np.testing.assert_array_equal(trend_index, truth_case)
        
    def test_hma_price_crossover_signal(self, series: pd.Series):
        ## generate the signals 
        signal = hma_price_crossover(self.series, 16)
        ## find the indices where signals = 1 or -1
        signal_index = signal.loc[signal!=0].index
        truth_case = signal_index
        ## find the trends from the indicator
        hull_ma = hma_price_crossover(self.series, 16)
        trend = np.sign(hull_ma - hull_ma.shift(1))
        trend = trend.fillna(0)
        ## find the indices where signals = 1 or -1
        trend_index = trend.loc[trend!=trend.shift(1)].index
        trend_index = trend_index.delete([0, 1])
        ##assert the two indices are equal
        np.testing.assert_array_equal(trend_index, truth_case)

    def test_hma_crossover_signal(self, series: pd.Series):
        ## generate the signals 
        signal = hma_crossover(self.series, 16, 81)
        ## find the indices where signals = 1 or -1
        signal_index = signal.loc[signal!=0].index
        truth_case = signal_index
        ## find the trends from the indicator
        hull_ma = hma_crossover(self.series, 16, 81)
        trend = np.sign(hull_ma - hull_ma.shift(1))
        trend = trend.fillna(0)
        ## find the indices where signals = 1 or -1
        trend_index = trend.loc[trend!=trend.shift(1)].index
        trend_index = trend_index.delete([0, 1])
        ##assert the two indices are equal
        np.testing.assert_array_equal(trend_index, truth_case)
  
	
  

    
if __name__ == '__main__':

    df = load_eod('AWU')
    
    # unit test for hma_trend_signal
    ## generate the signals 
    signal = hma_trend_signal(df.close)
    print("signal:", signal, '\n')
    ## find the indices where signals = 1 or -1
    signal_series = pd.Series(signal)
    signal_index = signal_series.loc[signal_series!=0].index
    print("signal_index:", signal_index, '\n')
    ## general the trends 
    hull_ma = pd.Series(numpy_matrix_hma(df.close.values, m=49))
    trend = np.sign(hull_ma - hull_ma.shift(1))
    trend = trend.fillna(0)
    print("trend:", trend, '\n')
    ## find the indices where signals = 1 or -1
    trend_index = trend.loc[trend!=trend.shift(1)].index
    trend_index = trend_index.delete([0, 1])
    print("trend_index:", trend_index, '\n')
    ##assert the two indices are equal
    assert trend_index.equals(signal_index), "Test Failed"
    assert np.array_equal(trend_index, signal_index), "Test Failed"
    TestSignals(df.close)
    unittest.main()