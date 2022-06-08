from vqti.load import load_eod
from vqti.profile import time_this
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List
import statistics


# TODO: DRY this up.

# 0th order is the price (position)
# 1st order is change in price (delta)
# 2nd order is change in change in price (delta^2 or acceleration)
# 3rd order etc ...

@time_this
def macd_python(close: List[float], n1: int = 2, n2: int = 3) -> List[float]:
	"""
	O(N)(3) Algorithm (?)

	The MACD result is a smoothed 1st order calculation expressed in dollars
	per share
	"""
	assert n1 < n2
	sma1 = [None] * (n1 - 1) # Units are dollars per share
	accum1 = sum(close[:n1])
	sma1.append(accum1 / n1)
	for i in range(n1, len(close)):
		accum1 += close[i]
		accum1 -= close[i - n1]
		sma1.append(accum1 / n1) # This is 0th order
	sma2 = [None] * (n2 - 1) # Units are dollars per share
	accum2 = sum(close[:n2])
	sma2.append(accum2 / n2)
	for i in range(n2, len(close)):
		accum2 += close[i]
		accum2 -= close[i - n2]
		sma2.append(accum2 / n2) # This is 0th order
	result = []
	for i in range(len(sma1)):
		if sma1[i] == None:
			result.append(None)
		elif sma2[i] == None:
			result.append(sma1[i])
		else:
			result.append(sma1[i] - sma2[i]) # Units are dollars per share, and its a 1st order calc
	return result

print(macd_python([1,2,3,4,5,6,7,8,9,10], n1=2, n2=3))

@time_this
def pandas_macd(close: pd.Series, n1: int = 2, n2: int = 3):
	assert n1 < n2
	sma1 = (close.cumsum() - close.cumsum().shift(n1)) / n1
	sma2 = (close.cumsum() - close.cumsum().shift(n2)) / n2
	return sma1 - sma2


@time_this
def numpy_macd(close: np.ndarray, n1: int = 2, n2: int = 3):
	assert n1 < n2
	sma1 = (close.cumsum()[n1:] - close.cumsum()[:-n1]) / n1
	sma2 = (close.cumsum()[n2:] - close.cumsum()[:-n2]) / n2
	return sma1 - sma2
	"""
	to do here:
	figure out how to deal with dimensional mismatch during subtraction
	"""

# TODO: Get numpy_macd to work by padding with np.nan
# TODO: Make sure your input dimensions match your output dimensions
# TODO [Optional]: Investigate numba

# Initial work on signals in Pure Python.  Pandas Below.
def calc_python_sma(close: List[float], m: int=10) -> List[float]:
	assert m >= 1, 'Window must be positive.'
	result = [None] * (m-1)
	for i in range(len(close)-(m-1)):
		result.append(sum(close[i:i+m])/m)
	assert len(result) == len(close), 'Result length does not match inputted data'
	return result

def calc_python_rolling_volatility(close: List[float], m: int):
	assert m >= 1, 'Window must be positive.'
	result = [None] * (m-1)
	for i in range((m-1),len(close)):
		#ignore None values for stdev
		if close[i-(m-1)] == None:
			result.append(None)
		else:
			result.append(statistics.stdev(close[i-(m-1):i+1])*((252/m)**.5))
	assert len(result) == len(close), 'Result length does not match inputted data'
	return result

def calc_python_macd(close: List[float], n1: int = 8, n2: int = 18) -> List[float]:
	assert n1 < n2, 'Second window length must be greater than first'
	sma1 = calc_python_sma(close, n1)
	sma2 = calc_python_sma(close, n2)
	macd = []
	for i in range(len(sma1)):
		if sma1[i] == None:
			macd.append(None)
		elif sma2[i] == None:
			macd.append(sma1[i])
		else:
			macd.append(sma1[i] - sma2[i])
	assert len(macd) == len(close), 'MACD length does not match inputted data'
	#Calculate rolling volatility over window n2
	vol = calc_python_rolling_volatility(macd, n2)
	#Divide by rolling volatility to standardize for signal line
	result = []
	for i in range(len(macd)):
		if vol[i] == None:
			result.append(None)
		else:
			result.append(macd[i] / vol[i])
	assert len(result) == len(close), 'Normalized MACD length does not match inputted data'
	return result

def calc_python_macd_signal(macd: List[float]) -> List[float]:
	result = []
	for i in range(len(macd) - 1):
		if macd[i] == None:
			result.append(None)
		elif macd[i] > 0 and macd[i+1] < 0:
			result.append(-1)
		elif macd[i] < 0 and macd[i+1] > 0:
			result.append(1)
		else:
			result.append(0)
	#Add an additional zero value for the final close value
	result.append(0)
	assert len(result) == len(macd), 'Signal length does not match MACD length'
	return result
# Pandas signal line-related functions

def calc_pandas_sma(close: pd.Series, m: int=10) -> pd.Series:
	assert m >=1, 'Window must be positive'
	result = close.rolling(m).mean()
	assert result.size == close.size, 'Result dimensions do not match'
	return result

def calc_pandas_rolling_volatility(prices, m):
	assert m >= 1, 'Window must be positive'
	result = prices.rolling(m).std() * ((252/m)**.5)
	assert result.size == prices.size, 'Result dimensions do not match'
	return result

def calc_pandas_macd(close: pd.Series, n1: int = 12, n2: int = 52) -> pd.Series:
	assert n1 < n2, 'Second window size must be greater than first'
	sma1 = calc_pandas_sma(close, n1)
	sma2 = calc_pandas_sma(close, n2)
	macd = sma1 - sma2
	#Divide by volatility
	vol = calc_pandas_rolling_volatility(macd, n2)
	result = macd / vol
	assert result.size == close.size, 'Result dimensions do not match'
	return result

def calc_pandas_macd_signal(close: pd.Series, n1: int = 5, n2: int = 34):
	sign = np.sign(calc_pandas_macd(close, n1, n2))
	shifted = sign.shift(1, axis = 0)
	result = sign * (sign != shifted)
	assert result.size == close.size, 'Result dimensions do not match'
	return result
# Haven't quite gotten this part working yet, but this should be another MACD signal line, calculated by taking the moving average of the MACD

def calc_pandas_macd_signal_v2(close: pd.Series, n1: int = 5, n2: int = 34, signal: int = 9):
	assert n1 < n2, 'Second window size must be greater than first'
	macd = calc_pandas_macd(close, n1, n2)
	macd_signal = calc_pandas_sma(macd, signal)
	# Divide by volatility
	vol = calc_pandas_rolling_volatility(macd, n2)
	macd_signal = macd_signal / vol
	macd_signal_shifted = macd_signal.shift(1, axis = 0)
	result = macd_signal * (macd_signal != macd_signal_shifted)
	return result


if __name__=="__main__":
	df = load_eod('AWU')
	print("macd pure python")
	macd_python(df.close.tolist(), n1=2, n2=3)
	print("macd pandas")
	pandas_macd(df.close, n1=2, n2=3)
	print("macd numpy")
	numpy_macd(df.close.values, n1=2, n2=3)


# TODO: Standardize this for the signal line by dividing by volatility

