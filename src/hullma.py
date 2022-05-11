from vqti.load import load_eod
from vqti.profile import time_this
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List
import math

# HMA= WMA(2*WMA(n/2) − WMA(n)),sqrt(n))
# recommended n = 4, 9, 16, 25, 49, 81

# @time_this
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
def numpy_wma(values: pd.Series, m: int) -> np.ndarray:
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

    # Front and back padding of series
    front_pad = max(m-1, 0)
   
    # Initialize the output array
    y = np.empty((n,))

    # Pad with na values
    y[:front_pad] = np.nan
    
    # Compute the moving average
    for i in range(front_pad, n):
        x = values[i]
        y[i] = weights.dot(values[(i-m+1):i+1])

    return y

@time_this
def numpy_hma(close: pd.Series, m: int=10) -> np.array:
	return numpy_wma((2* numpy_wma(close, int(m/2))) - (numpy_wma(close, m)), int(np.sqrt(m)))

@time_this
#fastest wma
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
	weights = []
	denom = (m * (m+1)) / 2
	for i in range(1, m+1):
		x = i / denom 
		weights.append(x)
	weights = np.array(weights)
	return close.rolling(window=m).apply(lambda x: np.sum(weights*x))

@time_this
# fastest hma
def pandas_hma(close: pd.Series, m: int=10) -> pd.Series:
	return pandas_wma((2* pandas_wma(close, int(m/2))) - (pandas_wma(close, m)), int(np.sqrt(m)))

def test_wma():
	ground_truth_result = [None, None, None, 3, 4, 5, 6, 7, 8, 9]
	test_result = pure_python_wma([1,2,3,4,5,6,7,8,9,10], 4) 
	assert len(ground_truth_result) == len(test_result)
	for i in range(len(ground_truth_result)):
		assert ground_truth_result[i] == test_result[i]


if __name__ == '__main__':
	
	df = load_eod('AWU')
	print(df)

	test_wma()
	result = pure_python_wma(df.close.tolist(), 4)
	result = pure_python_hma(df.close.tolist(), 4)
	result = numpy_wma(df.close.to_numpy(), 4)
	result = numpy_hma(df.close, 4)
	result = pandas_wma(df.close, 4)
	result = pandas_wma_2(df.close, 4)
	result = pandas_wma_3(df.close, 4)
	result = pandas_hma(df.close, 4)
	# print(result)