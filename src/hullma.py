from vqti.load import load_eod
from vqti.profile import time_this
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List
import math

# HMA= WMA(2*WMA(n/2) − WMA(n)),sqrt(n))
# recommended n = 4, 9, 16, 25, 49, 81

@time_this
def pure_python_wma(values: List[float], m: int=10)-> List[float]:
	"""
	This is O(nm).

	There is no O(n) solution to this algorithm, which is true of all filter
	operations.

	TODO: Investigate np.dot and other methods in fast python section 2.4
	"""

	# Building a triangular filter/weight/convolution array
	#       x
	#     x x
	#   x x x
	# x x x x
	# t ---->
	weighting = []
	for i in range(1, m+1):
		weighting.append(i)

	# Initial values
	moving_average = [None] * (m-1)
	
	# Apply it
	for i in range(m-1, len(values)):
		the_average = np.average(values[(i-m+1):i+1], weights=weighting)
		moving_average.append(the_average)

	return moving_average


@time_this
def pure_python_hma(close: List[float], m: int=10) -> List[float]:
	wma1 = np.array(pure_python_wma(close, int(m/2)))
	# multiply wma1 by 2 while keeping nan values
	wma1_multiplied = [None] * (int(m/2) -1)
	for i in range((int(m/2)-1), len(wma1)):
		y = wma1[i] *2
		wma1_multiplied.append(y)
  
	wma2 = np.array(pure_python_wma(close, m))
	# subtract wma2 from wma1 multiplied while keeping null values
	raw_hma = [0] * (m-1)
	for i in range((m-1), len(wma2)):
		raw_hma.append(np.subtract(wma1_multiplied[i], wma2[i]))
	hma= pure_python_wma(raw_hma, int(np.sqrt(m)))
	hma[0:m] = [None] * m
	return hma

@time_this
def pandas_wma(close: pd.Series, m: int=10) -> pd.Series:
	# TODO: Initialize the weights outside of the apply function
	if m > len(close):
		return [None] * len(close)
	return close.rolling(m).apply(lambda x: ((np.arange(m)+1)*x).sum()/(np.arange(m)+1).sum(), raw=True)

@time_this  
def pandas_wma_2(close: pd.Series, m: int=10) -> pd.Series:
	weights = []
	for i in range(1, m+1):
		weights.append(i)
	sum_weights = np.sum(weights)
	return close.rolling(window=m).apply(lambda x: np.sum(weights*x) / sum_weights)

@time_this
def pandas_wma_3(close: pd.Series, m: int=10) -> pd.Series:
	weights = np.array([])
	denom = (m * (m+1)) / 2
	for i in range(1, m+1):
		x = i / denom 
		weights.append(x)
	return close.rolling(window=m).apply(lambda x: np.sum(weights*x))

@time_this
def pandas_hma(close: pd.Series, m: int=10):
	return pandas_wma((2* pandas_wma(close, int(m/2))) - (pandas_wma(close, m)), int(np.sqrt(m)))



if __name__ == '__main__':
	
	df = load_eod('AWU')
	print(df)

	result = pure_python_wma(df.close.tolist(), 4)
	

	
	
	result = pandas_hma(df.close, 4)
	print(result)