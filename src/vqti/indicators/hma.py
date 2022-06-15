import pandas as pd
import numpy as np
from typing import List
# Hull Moving Average
# HMA = WMA(2*WMA(n/2) − WMA(n)),sqrt(n))
# recommended m = 4, 9, 16, 25, 49, 81

# fastest wma
def calculate_numpy_matrix_wma(values: np.ndarray, m: int=16) -> np.ndarray:
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
    wma = np.empty((n,))

    # Pad with na values
    wma[:front_pad] = np.nan

    # Build a matrix to multiply with weight vector
    q = np.empty((n - front_pad, m))
    for j in range(m):
        q[:,j] = values[j:(j+n-m+1)]

    wma[front_pad: len(values)] = q.dot(weights)

    return wma

# fastest hma
def calculate_numpy_matrix_hma(values: np.ndarray, m: int=10) -> np.ndarray:
	assert m >= 1, 'Period must be a positive integer'
	assert type(m) is int, 'Period must be a positive integer'
	assert len(values) >= m, 'Values must be >= period m'
	return calculate_numpy_matrix_wma(
                (2* calculate_numpy_matrix_wma(values, int(m/2))) -\
                    (calculate_numpy_matrix_wma(values, m)), int(np.sqrt(m)))

# fastest pandas wma
def calculate_pandas_wma(values: pd.Series, m: int=10) -> pd.Series:
	assert m >= 1, 'period must be a positive integer'
	assert type(m) is int, 'Period must be a positive integer'
	assert len(values) >= m, 'Values must be >= period m'
	return values.rolling(m).apply(lambda x: ((np.arange(m)+1)*x).sum()/ \
                            (np.arange(m)+1).sum(), raw=True)
 
def calculate_pandas_hma(values: pd.Series, m: int=10) -> pd.Series:
	assert m >= 1, 'Period must be a positive integer'
	assert type(m) is int, 'Period must be a positive integer'
	assert len(values) >= m, 'Values must be >= period m'
	return calculate_pandas_wma(2* calculate_pandas_wma(values, int(m/2)) - \
                        (calculate_pandas_wma(values, m)), int(np.sqrt(m)))

def _calculate_pure_python_wma(values: List[float], m: int=10)-> List[float]:
	assert m >= 1, 'Period must be a positive integer'
	assert type(m) is int, 'Period must be a positive integer'
	assert len(values) >= m, 'Values must be >= period m'
	# Building a triangular weight array
	weighting = []
	for i in range(1, m+1):
		weighting.append(i)

	# Initial values
	moving_average = [np.nan] * (m-1)
	
	# Apply weights
	for i in range(m-1, len(values)):
		the_average = np.average(values[(i-m+1):i+1], weights=weighting)
		moving_average.append(the_average)

	return moving_average


def _calculate_pure_python_hma(values: List[float], m: int=16) -> List[float]:
	assert m >= 1, 'Period must be a positive integer'
	assert type(m) is int, 'Period must be a positive integer'
	assert len(values) >= m, 'Values must be >= period m'
 
	wma1 = np.array(_calculate_pure_python_wma(values, int(m/2)))
	# multiply wma1 by 2 while keeping nan values
	wma1_multiplied = [None] * (int(m/2) -1)
	for i in range((int(m/2)-1), len(wma1)):
		y = wma1[i] *2
		wma1_multiplied.append(y)
  
	wma2 = np.array(_calculate_pure_python_wma(values, m))
	# subtract wma2 from wma1 multiplied while keeping null values
	raw_hma = [0] * (m-1)
	for i in range((m-1), len(wma2)):
		raw_hma.append(np.subtract(wma1_multiplied[i], wma2[i]))
	hma= _calculate_pure_python_wma(raw_hma, int(np.sqrt(m)))
	hma[0:m] = [None] * m
	return hma

def _calculate_numpy_wma(values: np.array, m: int=10) -> np.array:
	assert m >= 1, 'Period must be a positive integer'
	assert type(m) is int, 'Period must be a positive integer'
	assert len(values) >= m, 'Values must be >= period m'
 
	n = values.shape[0]
	weights = []
	denom = (m * (m+1)) / 2
	for i in range(1, m+1):
		x = i / denom 
		weights.append(x)

	weights = np.array(weights)
    # Exit early if m greater than length of values
	if m > n:
		return np.array([np.nan]*n)

    # Front padding of series
	front_pad = max(m-1, 0)
   
    # Initialize the output array
	wma = np.empty((n,))

    # Pad with na values
	wma[:front_pad] = np.nan
    
    # Compute the moving average
	for i in range(front_pad, n):
		x = values[i]
		wma[i] = weights.dot(values[(i-m+1):i+1])

	return wma


def _calculate_numpy_hma(values: np.ndarray, m: int=16) -> np.array:
	assert m >= 1, 'Period must be a positive integer'
	assert type(m) is int, 'Period must be a positive integer'
	assert len(values) >= m, 'Values must be >= period m'
	return _calculate_numpy_wma((2* _calculate_numpy_wma(values, int(m/2))) -\
                    (_calculate_numpy_wma(values, m)), int(np.sqrt(m)))

  
def _calculate_pandas_wma_2(values: pd.Series, m: int=10) -> pd.Series:
	assert m >= 1, 'Period must be a positive integer'
	assert type(m) is int, 'Period must be a positive integer'
	assert len(values) >= m, 'Values must be >= period m'
	weights = []
	for i in range(1, m+1):
		weights.append(i)
	sum_weights = np.sum(weights)
	return values.rolling(window=m).apply(lambda x: np.sum(weights*x) / \
                                       sum_weights)


def _calculate_pandas_wma_3(values: pd.Series, m: int=10) -> pd.Series:
	assert m >= 1, 'Period must be a positive integer'
	assert type(m) is int, 'Period must be a positive integer'
	assert len(values) >= m, 'Values must be >= period m'
	weights = []
	denom = (m * (m+1)) / 2
	for i in range(1, m+1):
		x = i / denom 
		weights.append(x)
	weights = np.array(weights)
	return values.rolling(window=m).apply(lambda x: np.sum(weights*x))

#############################
######## SIGNALS ############
#############################

def calculate_hma_trend_signal(series: pd.Series, m: int=49) -> pd.Series:
	hull_ma = pd.Series(calculate_numpy_matrix_hma(series.values, m))
	trend = np.sign(hull_ma - hull_ma.shift(1))
	signal = np.where(trend > trend.shift(1), 1, 0)
	signal = np.where(trend < trend.shift(1), -1, signal)
	return signal

def calculate_wma_trend_signal(series: pd.Series, m: int=30) -> pd.Series:
    wma = pd.Series(calculate_numpy_matrix_wma(series.values, m))
    trend = np.sign(wma - wma.shift(1))
    signal = np.where(trend > trend.shift(1), 1, 0)
    signal = np.where(trend < trend.shift(1), -1, signal)
    return signal

def calculate_rolling_volatility(series, m):
    return series.rolling(m).std() * np.sqrt(252/m)
 
def calculate_hma_zscore(series: pd.Series, m1: int=16, 
                         m2: int=81) -> pd.Series:
	assert m1 < m2, "m1 must be less than m2"
	assert m1 >= 1 and m2 >=1, 'Period must be a positive integer'
	assert type(m1) and type(m2) is int, 'Period must be a positive integer'
	assert len(series) >= m2, 'Values must be >= period m2'
	hma1 = pd.Series(calculate_numpy_matrix_hma(series.values, m1), 
                    index=series.index)
	hma2 = pd.Series(calculate_numpy_matrix_hma(series.values, m2), 
                    index=series.index) 
	vol = calculate_rolling_volatility(series, m2)
	return (hma1 - hma2) / vol 

def calculate_hma_zscore_signal(series: pd.Series, m1: int=16, m2: int=81):
    zscore = calculate_hma_zscore(series, m1, m2)
    zscore_sign = np.sign(zscore)
    zscore_shifted_sign = zscore_sign.shift(1, axis=0)
    return zscore_sign * (zscore_sign != zscore_shifted_sign)

def calculate_hma_macd_signal(series: pd.Series, m1: int=16, m2: int=49, 
                              sig: int=9) -> pd.Series:
	assert m1 < m2, "m1 must be less than m2"
	assert sig < m1, 'signal line must be less than m1'
	assert m1 >= 1 and m2 >=1, 'Period must be a positive integer'
	assert type(m1) and type(m2) is int, 'Period must be a positive integer'
	assert len(series) >= m2, 'Values must be >= period m2'
	hma1 = pd.Series(calculate_numpy_matrix_hma(series.values, m1),
                    index=series.index) 
	hma2 = pd.Series(calculate_numpy_matrix_hma(series.values, m2), 
                    index=series.index)
	macd = hma1 - hma2
	macd_sig = pd.Series(calculate_numpy_matrix_hma(macd.values, sig), 
                        index=series.index)
	hist = macd - macd_sig
	hist_sign = np.sign(hist)
	hist_shifted_sign = hist_sign.shift(1, axis=0)
	return hist_sign * (hist_sign != hist_shifted_sign)

def calculate_hma_price_crossover_signal(series: pd.Series, m: int=16):
    series = np.array(series)
    hull_ma = pd.Series(calculate_numpy_matrix_hma(series, m))
    sign = np.where(hull_ma > series, 1, 0)
    sign = pd.Series(np.where(hull_ma < series, -1, sign))
    price_crossover = np.where(sign > sign.shift(1), 1, 0)
    price_crossover = np.where(sign < sign.shift(1), -1, price_crossover)
    return price_crossover 

def calculate_hma_crossover_signal(series: pd.Series, m1: int=16, 
                                   m2: int=81)-> pd.Series:
    fast_hma = pd.Series(calculate_numpy_matrix_hma(series, m1))
    slow_hma = pd.Series(calculate_numpy_matrix_hma(series, m2))
    sign = np.sign(fast_hma - slow_hma)
    crossover = np.where(sign > sign.shift(1), 1, 0)
    crossover = np.where(sign < sign.shift(1), -1, crossover)
    return crossover
