from vqti.load import load_eod
from vqti.profile import time_this
import pandas as pd
import numpy as np
from typing import List
import statistics


#############################
### PYTHON IMPLEMENTATION ###
#############################

def calc_python_sma(close: List[float], m: int=10) -> List[float]:
	assert m >= 1, 'Window must be positive.'
	result = [None] * (m-1)
	for i in range(len(close)-(m-1)):
		result.append(sum(close[i:i+m])/m)
	assert len(result) == len(close), 'Result length does not match input data'
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
	assert len(result) == len(close), 'Result length does not match input data'
	return result

def calc_python_macd(close: List[float], n1: int = 8, n2: int = 18) \
		-> List[float]:
	assert n1 < n2, 'Second window length must be greater than first'
	sma1 = calc_python_sma(close, n1)
	sma2 = calc_python_sma(close, n2)
	macd = []
	for i in range(len(sma1)):
		if sma1[i] == None or sma2[i] == None:
			macd.append(None)
		else:
			macd.append(sma1[i] - sma2[i])
	assert len(macd) == len(close), 'MACD length does not match inputted data'
	return macd

def normalize_python_macd(macd: List, n2: int = 18) -> List:
	#Calculate rolling volatility over window n2
	vol = calc_python_rolling_volatility(macd, n2)
	#Divide by rolling volatility to standardize for signal line
	result = []
	for i in range(len(macd)):
		if vol[i] == None:
			result.append(None)
		else:
			result.append(macd[i] / vol[i])
	assert len(result) == len(macd), \
		'Normalized MACD length does not match input data'
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

#############################
### PANDAS IMPLEMENTATION ###
#############################

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
	return macd

def normalize_pandas_macd(macd: pd.Series, n2: int = 52) -> pd.Series:
	#Divide by volatility
	vol = calc_pandas_rolling_volatility(macd, n2)
	result = macd / vol
	assert result.size == macd.size, 'Result dimensions do not match'
	return result

def calc_pandas_macd_signal(close: pd.Series, n1: int = 5, n2: int = 34):
	macd = calc_pandas_macd(close, n1, n2)
	norm_macd = normalize_pandas_macd(macd, n2)
	sign = np.sign(norm_macd)
	shifted = sign.shift(1, axis = 0)
	result = sign * (sign != shifted)
	assert result.size == close.size, 'Result dimensions do not match'
	return result

# Haven't quite gotten this part working yet, but this should be another
# MACD signal line, calculated by taking the moving average of the MACD
def calc_pandas_macd_signal_v2(
		close: pd.Series, n1: int = 5, n2: int = 34, signal: int = 9
):
	assert n1 < n2, 'Second window size must be greater than first'
	macd = calc_pandas_macd(close, n1, n2)
	macd_signal = calc_pandas_sma(macd, signal)
	# Divide by volatility
	vol = calc_pandas_rolling_volatility(macd, n2)
	macd_signal = macd_signal / vol
	macd_signal_shifted = macd_signal.shift(1, axis = 0)
	result = macd_signal * (macd_signal != macd_signal_shifted)
	return result

######################
### TESTS & TIMING ###
######################

def _sma_test_case():
	print("Testing SMA functions")
	close: List = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	expected_sma_2day: List = [None,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5]

	# Test Python implementatation
	python_sma_2day: List = calc_python_sma(close, m =2)
	assert len(expected_sma_2day) == len(python_sma_2day)
	for idx, val in enumerate(python_sma_2day):
		assert val == expected_sma_2day[idx]

	# Test Pandas implementation
	pandas_sma_2day: pd.Series = calc_pandas_sma(pd.Series(close), m=2)
	assert pandas_sma_2day.equals(pd.Series(expected_sma_2day))

def _macd_test_case():
	print("Testing MACD function")
	close: List = [1, 2, 3, 4, 5, 6, 7, 8, 9, 14]
	expected_macd: List = [None, None, None, 1, 1, 1, 1, 1, 1, 2]

	python_macd: List = calc_python_macd(close, 2, 4)
	assert len(expected_macd) == len(python_macd)
	for idx, val in enumerate(python_macd):
		assert val == expected_macd[idx]

	pandas_macd: pd.Series = calc_pandas_macd(pd.Series(close), 2, 4)
	assert pandas_macd.equals(pd.Series(expected_macd))

def _normalization_test_case():
	print("Test functions normalizing MACD by volatilty")
	close: List = [1, 2, 3, 4, 6, 6, 2, 8, 9, 14]
	# Test Python implementation:
	python_macd: List = calc_python_macd(close, 2, 4)
	normalized_python_macd: List = normalize_python_macd(python_macd, 4)
	assert len(normalized_python_macd) == len(close)
	# Test Pandas implementation:
	pd_close = pd.Series(close)
	pandas_macd: pd.Series = calc_pandas_macd(pd_close, 2, 4)
	normalized_pandas_macd= normalize_pandas_macd(pandas_macd, 4)
	assert len(normalized_pandas_macd) == len(pd_close)

@time_this
def _time_python_macd_and_signal(close: List):
	macd = calc_python_macd(close, 12, 26)
	norm_macd = normalize_python_macd(macd, 26)
	return calc_python_macd_signal(norm_macd)

@time_this
def _time_pandas_macd_and_signal(close: pd.Series):
	return calc_pandas_macd_signal(close, 12, 26)


if __name__ == "__main__":
	_sma_test_case()
	_macd_test_case()
	_normalization_test_case()

	df: pd.DataFrame = load_eod('AWU')
	print("Timing Python implementation of MACD signal line")
	_time_python_macd_and_signal(df.close.to_list())
	print("Timing Pandas implementation of MACD signal line")
	_time_pandas_macd_and_signal(df.close)
