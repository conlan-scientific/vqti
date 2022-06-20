import pandas as pd
import numpy as np
from typing import List, Dict
from hma import calculate_numpy_matrix_hma, calculate_numpy_matrix_wma
# from vqti.load import load_eod #why cant I get this to work

def calculate_hma_trend_signal(series: pd.Series, m: int=49) -> pd.Series:
	hull_ma = calculate_numpy_matrix_hma(series, m)
	trend = np.sign(hull_ma - hull_ma.shift(1))
	signal = np.where(trend > trend.shift(1), 1, 0)
	signal = np.where(trend < trend.shift(1), -1, signal)
	return pd.Series(signal, index=series.index, dtype='float64', 
                  	 name='hma_trend_signal')

def calculate_wma_trend_signal(series: pd.Series, m: int=30) -> pd.Series:
    wma = calculate_numpy_matrix_wma(series, m)
    trend = np.sign(wma - wma.shift(1))
    signal = np.where(trend > trend.shift(1), 1, 0)
    signal = np.where(trend < trend.shift(1), -1, signal)
    return pd.Series(signal, index=series.index, dtype='float64', 
                     name='wma_trend_signal')


def calculate_rolling_volatility(series: pd.Series, m: int) -> pd.Series:
    volatility = series.rolling(m).std() * np.sqrt(252/m)
    return pd.Series(volatility, index=series.index, dtype='float64', 
                  	 name='rolling_volatility')
    
def calculate_hma_zscore(series: pd.Series, m1: int=16, 
                         m2: int=81) -> pd.Series:
	assert m1 < m2, "m1 must be less than m2"
	assert m1 >= 1 and m2 >=1, 'Period must be a positive integer'
	assert type(m1) and type(m2) is int, 'Period must be a positive integer'
	assert len(series) >= m2, 'Values must be >= period m2'
	hma1 = pd.Series(calculate_numpy_matrix_hma(series, m1), 
                    index=series.index)
	hma2 = pd.Series(calculate_numpy_matrix_hma(series, m2), 
                    index=series.index) 
	vol = calculate_rolling_volatility(series, m2)
	zscore = (hma1 - hma2) / vol 
	return pd.Series(zscore.values, index=series.index, dtype='float64', 
                	 name='hma_zscore')

def calculate_hma_zscore_signal(series: pd.Series, m1: int=16, 
                                m2: int=81) -> pd.Series:
	zscore = calculate_hma_zscore(series, m1, m2)
	zscore_sign = np.sign(zscore)
	zscore_shifted_sign = zscore_sign.shift(1, axis=0)
	signal = zscore_sign * (zscore_sign != zscore_shifted_sign)
	return pd.Series(signal, index=series.index, dtype='float64', 
                     name='hma_zscore_signal')


def calculate_hma_macd_signal(series: pd.Series, m1: int=16, m2: int=49, 
                              sig: int=9) -> pd.Series:
	assert m1 < m2, "m1 must be less than m2"
	assert sig < m1, 'signal line must be less than m1'
	assert m1 >= 1 and m2 >=1, 'Period must be a positive integer'
	assert type(m1) and type(m2) is int, 'Period must be a positive integer'
	assert len(series) >= m2, 'Values must be >= period m2'
	hma1 = pd.Series(calculate_numpy_matrix_hma(series, m1),
                    index=series.index) 
	hma2 = pd.Series(calculate_numpy_matrix_hma(series, m2), 
                    index=series.index)
	macd = hma1 - hma2
	macd_sig = pd.Series(calculate_numpy_matrix_hma(macd, sig), 
                        index=series.index)
	hist = macd - macd_sig
	hist_sign = np.sign(hist)
	hist_shifted_sign = hist_sign.shift(1, axis=0)
	signal = hist_sign * (hist_sign != hist_shifted_sign)
	return pd.Series(signal, index=series.index, dtype='float64', 
                  	 name='hma_macd_signal')

def calculate_hma_price_crossover_signal(series: pd.Series, 
                                         m: int=16) -> pd.Series:
    hull_ma = calculate_numpy_matrix_hma(series, m)
    sign = np.where(hull_ma > series, 1, 0)
    sign = pd.Series(np.where(hull_ma < series, -1, sign))
    price_crossover = np.where(sign > sign.shift(1), 1, 0)
    price_crossover = np.where(sign < sign.shift(1), -1, price_crossover)
    return pd.Series(price_crossover, index=series.index, dtype='float64', 
                     name='hma_price_crossover')


def calculate_hma_crossover_signal(series: pd.Series, m1: int=16, 
                                   m2: int=81)-> pd.Series:
    fast_hma = calculate_numpy_matrix_hma(series, m1)
    slow_hma = calculate_numpy_matrix_hma(series, m2)
    sign = np.sign(fast_hma - slow_hma)
    crossover = np.where(sign > sign.shift(1), 1, 0)
    crossover = np.where(sign < sign.shift(1), -1, crossover)
    return pd.Series(crossover, index=series.index, dtype='float64', 
                     name='hma_crossover_signal')
    

if __name__ == '__main__':
     
    df = load_eod('AWU')
    
    # unit test for hma_trend_signal
    ## generate the signals 
    signal = calculate_hma_trend_signal(df.close)
    print("signal:", signal, '\n')
    ## find the indices where signals = 1 or -1
    signal_series = pd.Series(signal)
    signal_index = signal_series.loc[signal_series!=0].index
    print("signal_index:", signal_index, '\n')
    ## general the trends 
    hull_ma = calculate_numpy_matrix_hma(df.close, m=49)
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
    